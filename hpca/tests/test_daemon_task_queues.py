from threading import Event

from hpca.orchestrator.daemon_tasks import (
    DaemonTaskScheduler,
    MaterialDesignTask,
    PreoptimizationTask,
)


def test_design_and_preoptimization_use_independent_queues(tmp_path):
    scheduler = DaemonTaskScheduler(design_workers=1, preoptimization_workers=1)
    design_started = Event()
    release_design = Event()
    preopt_finished = Event()

    def blocked_design():
        design_started.set()
        assert release_design.wait(timeout=2)

    try:
        design = scheduler.submit(MaterialDesignTask(tmp_path, blocked_design))
        assert design_started.wait(timeout=1)
        preopt = scheduler.submit(PreoptimizationTask(tmp_path, preopt_finished.set))
        assert preopt_finished.wait(timeout=1), "preopt was serialized behind design"
        release_design.set()
        design.result(timeout=1)
        preopt.result(timeout=1)
    finally:
        release_design.set()
        scheduler.shutdown()


def test_each_queue_enforces_its_worker_limit(tmp_path):
    scheduler = DaemonTaskScheduler(design_workers=1, preoptimization_workers=1)
    first_started = Event()
    release_first = Event()
    second_started = Event()

    def first():
        first_started.set()
        assert release_first.wait(timeout=2)

    try:
        future1 = scheduler.submit(MaterialDesignTask(tmp_path / "one", first))
        assert first_started.wait(timeout=1)
        future2 = scheduler.submit(MaterialDesignTask(tmp_path / "two", second_started.set))
        assert not second_started.wait(timeout=0.1)
        release_first.set()
        future1.result(timeout=1)
        future2.result(timeout=1)
        assert second_started.is_set()
    finally:
        release_first.set()
        scheduler.shutdown()


def test_worker_counts_must_be_positive():
    try:
        DaemonTaskScheduler(design_workers=0, preoptimization_workers=1)
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("zero design workers should be rejected")
