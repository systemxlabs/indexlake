pub mod data;

#[macro_export]
macro_rules! benchprintln {
    ($($arg:tt)*) => {{
        println!("benchmark: {}", format_args!($($arg)*));
    }};
}

pub fn bench_fast_mode_enabled() -> bool {
    matches!(
        std::env::var("INDEXLAKE_BENCH_FAST").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}
