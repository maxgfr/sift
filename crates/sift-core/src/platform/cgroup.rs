//! Memory limits imposed by Linux control groups.
//!
//! A container started with `docker run --memory=2g` on a 64 GB host reads 64 GB from
//! `/proc/meminfo`: the kernel reports the machine, not the box the process was put in.
//! Plan a fit against that and the model is OOM-killed while loading. The binding figure
//! is the cgroup's, and this module finds it.
//!
//! Written against a root directory rather than `/`, so the reader is tested on fixture
//! trees on every OS — none of which needs to be Linux, or inside a container.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::path::{Path, PathBuf};

/// Limits at or above this are "no limit": cgroup v1 writes `PAGE_COUNTER_MAX`, a number
/// near 2^63, rather than a word.
const UNLIMITED_FROM: u64 = 1 << 62;

/// The memory limit binding this process, and what it is using of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgroupMemory {
    /// The tightest limit on the way from this process's cgroup to the root.
    pub limit: u64,
    /// Bytes charged to this process's cgroup now, when the kernel says.
    pub usage: Option<u64>,
}

impl CgroupMemory {
    /// What a new allocation could still have before the limit binds.
    pub fn headroom(&self) -> Option<u64> {
        self.usage.map(|u| self.limit.saturating_sub(u))
    }
}

/// Find the memory limit for this process under `root` (`/` in production).
///
/// Handles both hierarchies. v2 names one unified cgroup (`0::/path`) whose `memory.max`
/// holds the limit; v1 names a `memory` controller path whose `memory.limit_in_bytes`
/// does. Either way every ancestor is checked too, because a parent's limit binds its
/// children — and inside a container the path the kernel names is the host's, which is
/// not mounted, so only the container's own root exists to be read.
pub fn memory_limit(root: &Path) -> Option<CgroupMemory> {
    let membership = std::fs::read_to_string(root.join("proc/self/cgroup")).ok()?;
    let fs = root.join("sys/fs/cgroup");

    for line in membership.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            if let Some(m) = walk(&fs, path, "memory.max", "memory.current") {
                return Some(m);
            }
        } else if controllers.split(',').any(|c| c == "memory") {
            if let Some(m) = walk(
                &fs.join("memory"),
                path,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            ) {
                return Some(m);
            }
        }
    }
    None
}

/// The tightest limit from `mount/path` up to `mount`, with the leaf-most usage found.
fn walk(mount: &Path, path: &str, limit_file: &str, usage_file: &str) -> Option<CgroupMemory> {
    let mut dir: PathBuf = mount.join(path.trim_start_matches('/'));
    let mut limit: Option<u64> = None;
    let mut usage: Option<u64> = None;
    loop {
        if let Some(l) = read_limit(&dir.join(limit_file)) {
            limit = Some(limit.map_or(l, |cur| cur.min(l)));
        }
        if usage.is_none() {
            usage = read_u64(&dir.join(usage_file));
        }
        if dir == mount || !dir.pop() || !dir.starts_with(mount) {
            break;
        }
    }
    limit.map(|limit| CgroupMemory { limit, usage })
}

/// A limit file's value, or `None` for "max", "unlimited" or a missing file.
fn read_limit(path: &Path) -> Option<u64> {
    read_u64(path).filter(|&v| v > 0 && v < UNLIMITED_FROM)
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, text: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    #[test]
    fn a_v2_container_limit_is_read_from_its_own_root() {
        // Inside a container with a cgroup namespace: `0::/`, and the limit at the root.
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "proc/self/cgroup", "0::/\n");
        write(d.path(), "sys/fs/cgroup/memory.max", "2147483648\n");
        write(d.path(), "sys/fs/cgroup/memory.current", "104857600\n");
        let m = memory_limit(d.path()).expect("limited");
        assert_eq!(m.limit, 2 << 30);
        assert_eq!(m.headroom(), Some((2 << 30) - (100 << 20)));
    }

    #[test]
    fn a_v2_parent_limit_binds_a_child_with_none() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "proc/self/cgroup", "0::/app.slice/job.scope\n");
        write(
            d.path(),
            "sys/fs/cgroup/app.slice/memory.max",
            "4294967296\n",
        );
        write(
            d.path(),
            "sys/fs/cgroup/app.slice/job.scope/memory.max",
            "max\n",
        );
        assert_eq!(memory_limit(d.path()).unwrap().limit, 4 << 30);
    }

    #[test]
    fn a_v1_limit_is_found_even_when_the_named_path_is_not_mounted() {
        // cgroup v1 in Docker: /proc/self/cgroup names the host's path, which the
        // container cannot see; its own cgroup is mounted at the controller root.
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "proc/self/cgroup",
            "12:cpu,cpuacct:/docker/abc\n7:memory:/docker/abc\n",
        );
        write(
            d.path(),
            "sys/fs/cgroup/memory/memory.limit_in_bytes",
            "2147483648\n",
        );
        write(
            d.path(),
            "sys/fs/cgroup/memory/memory.usage_in_bytes",
            "1048576\n",
        );
        let m = memory_limit(d.path()).expect("limited");
        assert_eq!(m.limit, 2 << 30);
        assert_eq!(m.usage, Some(1 << 20));
    }

    #[test]
    fn no_limit_is_none_rather_than_a_huge_number() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "proc/self/cgroup", "0::/user.slice\n");
        write(d.path(), "sys/fs/cgroup/user.slice/memory.max", "max\n");
        assert_eq!(memory_limit(d.path()), None);

        let v1 = tempfile::tempdir().unwrap();
        write(v1.path(), "proc/self/cgroup", "4:memory:/\n");
        write(
            v1.path(),
            "sys/fs/cgroup/memory/memory.limit_in_bytes",
            "9223372036854771712\n",
        );
        assert_eq!(
            memory_limit(v1.path()),
            None,
            "PAGE_COUNTER_MAX is not a limit"
        );
    }

    #[test]
    fn a_machine_without_cgroups_is_none() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(memory_limit(d.path()), None);
    }
}
