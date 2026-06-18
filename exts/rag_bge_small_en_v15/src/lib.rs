mod chunk;
mod errors;
mod embeddings {
    tonic::include_proto!("embeddings");
}
mod util;

use embeddings::{
    embedding_generator_server::{EmbeddingGenerator, EmbeddingGeneratorServer},
    EmbeddingReply, EmbeddingRequest,
};
use errors::*;
use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
#[cfg(feature = "remote_onnx")]
use futures_util::StreamExt;
use pgrx::{bgworkers::*, prelude::*};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs, os::unix::fs::PermissionsExt, sync::OnceLock};
use tokio::{
    net::UnixListener,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{transport::Server, Request, Response, Status};

// macros

mconst!(ext_name, "rag_bge_small_en_v15");
mconst!(model_path, "../../../lib/bge_small_en_v15/");

#[cfg(feature = "remote_onnx")]
const ONNX_SIZE: usize = 133_093_490;

macro_rules! socket_path {
    ($pid:expr) => {
        format!(concat!("/tmp/.s.pgrag.", ext_name!(), ".{}"), $pid)
    };
}

// init

pg_module_magic!();

static PID: OnceLock<i64> = OnceLock::new();
static TEXT_EMBEDDING: tokio::sync::OnceCell<TextEmbedding> = tokio::sync::OnceCell::const_new();

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    let pid = std::process::id() as i64;
    PID.set(pid)
        .expect_or_pg_err("Impossible concurrent access to set PID value");

    BackgroundWorkerBuilder::new(concat!(ext_name!(), " embeddings background worker"))
        .set_function("background_main")
        .set_library(ext_name!())
        .set_argument(pid.into_datum())
        .set_restart_time(Some(Duration::from_secs(1)))
        .enable_spi_access()
        .load();
}

// model loading

#[cfg(not(feature = "remote_onnx"))]
async fn get_onnx() -> Result<Vec<u8>, reqwest::Error> {
    Ok(include_bytes!(concat!(model_path!(), "model.onnx")).to_vec())
}

#[cfg(feature = "remote_onnx")]
async fn get_onnx() -> Result<Vec<u8>, reqwest::Error> {
    let url = env!("REMOTE_ONNX_URL");
    let response = reqwest::get(url).await?;
    let mut stream = response.bytes_stream();
    let mut vec: Vec<u8> = Vec::with_capacity(ONNX_SIZE);
    while let Some(chunk) = stream.next().await {
        vec.extend(chunk?);
    }
    Ok(vec)
}

// background worker

pub struct EmbeddingGeneratorStruct {
    thread_pool: ThreadPool,
}

#[tonic::async_trait]
impl EmbeddingGenerator for EmbeddingGeneratorStruct {
    async fn get_embedding(&self, request: Request<EmbeddingRequest>) -> Result<Response<EmbeddingReply>, Status> {
        let text = request.into_inner().text;
        let model = match TEXT_EMBEDDING
            .get_or_try_init(|| async {
                let onnx_file = get_onnx().await?;
                let tokenizer_files = TokenizerFiles {
                    tokenizer_file: include_bytes!(concat!(model_path!(), "tokenizer.json")).to_vec(),
                    config_file: include_bytes!(concat!(model_path!(), "config.json")).to_vec(),
                    special_tokens_map_file: include_bytes!(concat!(model_path!(), "special_tokens_map.json")).to_vec(),
                    tokenizer_config_file: include_bytes!(concat!(model_path!(), "tokenizer_config.json")).to_vec(),
                };
                let user_def_model = UserDefinedEmbeddingModel {
                    onnx_file,
                    tokenizer_files,
                };
                TextEmbedding::try_new_from_user_defined(user_def_model, Default::default())
            })
            .await
        {
            Err(err) => return Err(Status::internal(err.to_string())),
            Ok(model) => model,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread_pool.spawn(|| {
            let embeddings = model.embed(vec![text], None);
            tx.send(embeddings).expect("Channel send failed");
        });

        match rx.await {
            Err(_) => Err(Status::internal("Embedding process crashed")),
            Ok(Err(embed_error)) => Err(Status::internal(embed_error.to_string())),
            Ok(Ok(embeddings)) => {
                let embedding = embeddings.into_iter().next().unwrap_or_pg_err("Empty result vector");
                let reply = EmbeddingReply { embedding };
                Ok(Response::new(reply))
            }
        }
    }
}

#[pg_guard]
#[no_mangle]
pub extern "C-unwind" fn background_main(arg: pg_sys::Datum) {
    let pid = unsafe { i64::from_polymorphic_datum(arg, false, pg_sys::INT8OID).unwrap_or_pg_err("No PID received") };
    let name = BackgroundWorker::get_name();
    log!("{ERR_PREFIX} {name} started, received PID {pid}");

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGTERM);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect_or_pg_err("Couldn't build tokio runtime for server")
        .block_on(async {
            unsafe { pg_sys::BackgroundWorkerBlockSignals() };
            let path = socket_path!(pid);
            fs::remove_file(&path).unwrap_or_default(); // it's not an error if the file isn't there
            let uds = UnixListener::bind(&path).expect_or_pg_err(&format!("Couldn't create socket at {}", &path));
            fs::set_permissions(&path, fs::Permissions::from_mode(0o777))
                .expect_or_pg_err(&format!("Couldn't set permissions for {}", &path));
            log!("{ERR_PREFIX} {} created socket {}", name, &path);

            let num_threads = match std::thread::available_parallelism() {
                Err(_) => 0, // automatic
                Ok(cpu_count) => match cpu_count.get() {
                    1 => 1,
                    cpus => cpus - 1,
                },
            };
            let embedder = EmbeddingGeneratorStruct {
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .expect_or_pg_err("Couldn't build thread pool"),
            };
            log!("{ERR_PREFIX} {} requested num_threads({})", name, num_threads);

            let uds_stream = UnixListenerStream::new(uds);
            Server::builder()
                .add_service(EmbeddingGeneratorServer::new(embedder))
                .serve_with_incoming_shutdown(uds_stream, async {
                    unsafe { pg_sys::BackgroundWorkerUnblockSignals() };
                    // wait_latch is not an async function and does not suspend
                    while BackgroundWorker::wait_latch(Some(Duration::from_secs(0))) {
                        unsafe { pg_sys::BackgroundWorkerBlockSignals() };
                        // suspend so that other asyncs/threads can run
                        sleep(Duration::from_millis(500)).await;
                        unsafe { pg_sys::BackgroundWorkerUnblockSignals() };
                    }
                })
                .await
                .expect_or_pg_err("Couldn't create server");
        });
}

// extension function(s)

#[pg_schema]
mod rag_bge_small_en_v15 {
    pub mod embeddings {
        tonic::include_proto!("embeddings");
    }

    use super::{errors::*, PID};
    use hyper_util::rt::TokioIo;
    use pgrx::prelude::*;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};
    use tower::service_fn;

    use embeddings::embedding_generator_client::EmbeddingGeneratorClient;
    use embeddings::EmbeddingRequest;

    #[pg_extern(immutable, strict)]
    pub fn _embedding(text: String) -> Vec<f32> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect_or_pg_err("Couldn't build tokio runtime for client")
            .block_on(async {
                let channel = Endpoint::try_from("http://[::]:80") // URL must be valid but is ignored
                    .expect_or_pg_err("Failed to create endpoint")
                    .connect_with_connector(service_fn(|_: Uri| async {
                        let pid = PID.get().unwrap_or_pg_err("Couldn't get PID");
                        let path = socket_path!(pid);
                        Ok::<_, std::io::Error>(TokioIo::new(
                            UnixStream::connect(&path)
                                .await
                                .expect_or_pg_err(&format!("Couldn't connect worker stream {}", &path)),
                        ))
                    }))
                    .await
                    .expect_or_pg_err("Couldn't connect worker channel");

                let mut client = EmbeddingGeneratorClient::new(channel);
                let request = tonic::Request::new(EmbeddingRequest { text });
                let response = client
                    .get_embedding(request)
                    .await
                    .expect_or_pg_err("Worker process returned error");

                response.into_inner().embedding
            })
    }

    extension_sql!(
        "CREATE FUNCTION rag_bge_small_en_v15.embedding_for_passage(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding(input)::vector(384);
        $$;
        CREATE FUNCTION rag_bge_small_en_v15.embedding_for_query(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding('Represent this sentence for searching relevant passages: ' || input)::vector(384);
        $$;",
        name = "embeddings",
    );
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_bge_small_en_v15::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_embedding_length() {
        assert_eq!(_embedding("hello world!".to_string()).len(), 384);
    }

    #[pg_test]
    fn test_embedding_immutability() {
        assert_eq!(_embedding("hello world!".to_string()), _embedding("hello world!".to_string()));
    }

    #[pg_test]
    fn test_embedding_variability() {
        assert_ne!(_embedding("hello world!".to_string()), _embedding("bye moon!".to_string()));
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
