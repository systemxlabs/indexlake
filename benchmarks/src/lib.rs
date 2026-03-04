pub mod data;

#[macro_export]
macro_rules! benchprintln {
    ($($arg:tt)*) => {{
        println!("benchmark: {}", format_args!($($arg)*));
    }};
}
