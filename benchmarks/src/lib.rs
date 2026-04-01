use std::time::Duration;

use indexlake::{ILError, ILResult, table::Table};

pub mod data;

#[macro_export]
macro_rules! benchprintln {
    ($($arg:tt)*) => {{
        println!("benchmark: {}", format_args!($($arg)*));
    }};
}

pub async fn wait_data_files_ready(
    table: &Table,
    expected_data_file_count: usize,
    timeout: Duration,
) -> ILResult<()> {
    let sleep_duration = Duration::from_secs(10);
    let mut total_wait_duration = Duration::from_secs(0);
    loop {
        let data_file_count = table.data_file_count().await?;
        if data_file_count == expected_data_file_count {
            break;
        }
        if total_wait_duration > timeout {
            benchprintln!(
                "Timeout waiting data files ready after {}s, last data file count: {data_file_count}, expected {expected_data_file_count}",
                timeout.as_secs()
            );
            return Err(ILError::internal("Timeout waiting data files ready"));
        }

        tokio::time::sleep(sleep_duration).await;
        total_wait_duration += sleep_duration;
    }

    Ok(())
}
