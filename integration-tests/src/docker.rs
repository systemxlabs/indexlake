use std::process::Command;

pub fn run_command(mut cmd: Command, desc: impl ToString) {
    let desc = desc.to_string();
    println!("Starting to {desc}, command: {cmd:?}");
    let exit = cmd.status().unwrap();
    if exit.success() {
        println!("{desc} succeed!")
    } else {
        panic!("{desc} failed: {exit:?}");
    }
}

pub fn get_cmd_output_result(mut cmd: Command, desc: impl ToString) -> Result<String, String> {
    let desc = desc.to_string();
    println!("Starting to {desc}, command: {cmd:?}");
    let result = cmd.output();
    match result {
        Ok(output) => {
            if output.status.success() {
                println!("{desc} succeed!");
                Ok(String::from_utf8(output.stdout).unwrap())
            } else {
                Err(format!("{desc} failed with rc: {:?}", output.status))
            }
        }
        Err(err) => Err(format!("{desc} failed with error: {err}")),
    }
}

pub fn get_cmd_output(cmd: Command, desc: impl ToString) -> String {
    let result = get_cmd_output_result(cmd, desc);
    match result {
        Ok(output_str) => output_str,
        Err(err) => panic!("{err}"),
    }
}

#[derive(Debug)]
pub struct DockerCompose {
    project_name: String,
    docker_compose_dir: String,
}

impl DockerCompose {
    pub fn new(project_name: impl ToString, docker_compose_dir: impl ToString) -> Self {
        Self {
            project_name: project_name.to_string(),
            docker_compose_dir: docker_compose_dir.to_string(),
        }
    }

    pub fn project_name(&self) -> &str {
        self.project_name.as_str()
    }

    fn get_os_arch() -> String {
        let mut cmd = Command::new("docker");
        cmd.arg("info")
            .arg("--format")
            .arg("{{.OSType}}/{{.Architecture}}");

        let result = get_cmd_output_result(cmd, "Get os arch".to_string());
        match result {
            Ok(value) => value.trim().to_string(),
            Err(_err) => {
                // docker/podman do not consistently place OSArch info in the same json path
                // across OS and versions Below tries an alternative path if the
                // above path fails
                let mut alt_cmd = Command::new("docker");
                alt_cmd
                    .arg("info")
                    .arg("--format")
                    .arg("{{.Version.OsArch}}");
                get_cmd_output(alt_cmd, "Get os arch".to_string())
                    .trim()
                    .to_string()
            }
        }
    }

    pub fn up(&self) {
        let mut cmd = Command::new("docker");
        cmd.current_dir(&self.docker_compose_dir);

        cmd.env("DOCKER_DEFAULT_PLATFORM", Self::get_os_arch());

        cmd.args(vec![
            "compose",
            "-p",
            self.project_name.as_str(),
            "up",
            "-d",
            "--wait",
            "--timeout",
            "1200000",
        ]);

        run_command(
            cmd,
            format!(
                "Starting docker compose in {}, project name: {}",
                self.docker_compose_dir, self.project_name
            ),
        )
    }

    pub fn down(&self) {
        let mut cmd = Command::new("docker");
        cmd.current_dir(&self.docker_compose_dir);

        cmd.args(vec![
            "compose",
            "-p",
            self.project_name.as_str(),
            "down",
            "-v",
            "--remove-orphans",
        ]);

        run_command(
            cmd,
            format!(
                "Stopping docker compose in {}, project name: {}",
                self.docker_compose_dir, self.project_name
            ),
        )
    }
}
