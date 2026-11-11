"""覆盖低内存模式：按名称卸载原子模型、Context 属性释放、CPU 会话内存池与默认线程数。"""

from __future__ import annotations

import contextlib
import gc
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import onnxruntime as ort
import pytest

from mineru.backend.analysis.pdf import window
from mineru.model.runtime import cpu_threads, hybrid, onnx
from mineru.model.runtime.contracts import AtomicModelName


@dataclass(frozen=True)
class _Decision:
    """替代真实设备决策；可哈希，作为缓存键的一部分。"""

    backend: str = "onnx"
    model_device: str = "cpu"


class _Model:
    """替代真实模型，便于用弱引用确认已被回收。"""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def atom_cache(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """隔离原子模型缓存，并以假模型替换真实初始化，返回每次创建的模型名称。"""
    created: list[str] = []
    monkeypatch.setattr(hybrid.AtomModelSingleton, "_models", {})
    monkeypatch.setattr(hybrid.AtomModelSingleton, "_model_names", {})
    monkeypatch.setattr(hybrid, "resolve_model_device", lambda *args, **kwargs: _Decision())
    monkeypatch.setattr(hybrid, "model_device_scope", lambda decision: contextlib.nullcontext())

    def fake_init(model_name: str, **kwargs: object) -> _Model:
        created.append(model_name)
        return _Model(model_name)

    monkeypatch.setattr(hybrid, "atom_model_init", fake_init)
    return created


def test_evict_removes_named_models_whatever_the_cache_key_shape(atom_cache: list[str]) -> None:
    """按名称卸载不能依赖缓存键的内部结构：4.2 把键改成 (设备决策, 模型键) 后旧解析会静默失配。"""
    manager = hybrid.AtomModelSingleton()
    layout = manager.get_atom_model(AtomicModelName.Layout)
    mfr = manager.get_atom_model(AtomicModelName.MFR)
    ocr = manager.get_atom_model(AtomicModelName.OCR)
    refs = [weakref.ref(layout), weakref.ref(mfr)]
    del layout, mfr

    assert manager.evict({AtomicModelName.Layout, AtomicModelName.MFR}) == 2
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert [hybrid.AtomModelSingleton._model_names[key] for key in manager._models] == [AtomicModelName.OCR]
    assert manager.get_atom_model(AtomicModelName.OCR) is ocr

    manager.get_atom_model(AtomicModelName.Layout)
    assert atom_cache == ["layout", "mfr", "ocr", "layout"]


def test_evict_of_a_model_never_loaded_is_a_no_op(atom_cache: list[str]) -> None:
    """未加载过的模型不在缓存中，卸载返回零且不触碰其它模型。"""
    manager = hybrid.AtomModelSingleton()
    manager.get_atom_model(AtomicModelName.OCR)
    assert manager.evict({AtomicModelName.MFR}) == 0
    assert len(manager._models) == 1


def _context_with(evicted: list[set[str]]) -> hybrid.HybridLocalModelContext:
    """构造不加载真实模型的 Context，只带缓存属性与记录卸载请求的管理器。"""
    context = object.__new__(hybrid.HybridLocalModelContext)
    context.atom_model_manager = SimpleNamespace(evict=lambda names: evicted.append(set(names)) or len(names))
    context.__dict__.update(layout_model="L", mfr_model="M", ocr_model="O")
    return context


def test_release_models_does_nothing_without_low_memory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认模式下阶段结束不卸载任何模型。"""
    monkeypatch.delenv("MINERU_LOW_MEMORY", raising=False)
    evicted: list[set[str]] = []
    context = _context_with(evicted)
    context.release_models(AtomicModelName.Layout, AtomicModelName.MFR)
    assert evicted == []
    assert {"layout_model", "mfr_model"} <= context.__dict__.keys()


def test_release_models_drops_attributes_and_evicts_in_low_memory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """低内存模式下移出 Context 缓存属性并按名称卸载，未点名的模型保留。"""
    monkeypatch.setenv("MINERU_LOW_MEMORY", "1")
    monkeypatch.setattr(hybrid, "trim_process_heap", lambda: True)
    evicted: list[set[str]] = []
    context = _context_with(evicted)
    context.release_models(AtomicModelName.Layout, AtomicModelName.MFR)
    assert evicted == [{AtomicModelName.Layout, AtomicModelName.MFR}]
    assert "layout_model" not in context.__dict__ and "mfr_model" not in context.__dict__
    assert context.__dict__["ocr_model"] == "O"


def test_stage_release_ignores_contexts_without_the_capability() -> None:
    """未实现 release_models 的 Context（如测试替身）不受影响。"""
    window._release_stage_models(SimpleNamespace(), AtomicModelName.MFR)


def test_cpu_sessions_drop_memory_pools_only_in_low_memory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """低内存模式关闭 CPU 会话的内存池与内存模式；默认保持 ORT 原有设置。"""
    opts = ort.SessionOptions()
    onnx.release_cpu_memory_between_runs(opts)
    assert opts.enable_cpu_mem_arena is False and opts.enable_mem_pattern is False

    created: list[ort.SessionOptions] = []
    monkeypatch.setattr(onnx.ort, "InferenceSession", lambda path, sess_options, providers: created.append(sess_options))
    monkeypatch.delenv("MINERU_LOW_MEMORY", raising=False)
    onnx.create_ort_session("model.onnx")
    monkeypatch.setenv("MINERU_LOW_MEMORY", "1")
    onnx.create_ort_session("model.onnx")
    assert created[0].enable_cpu_mem_arena is True
    assert created[1].enable_cpu_mem_arena is False and created[1].enable_mem_pattern is False


@pytest.mark.parametrize(("cpus", "expected"), [(1, 1), (2, 2), (4, 4), (16, 4)])
def test_default_intra_op_threads_follow_the_cpus_available(monkeypatch: pytest.MonkeyPatch, cpus: int, expected: int) -> None:
    """未显式配置时 ONNX 算子内线程数取 min(4, 可用 CPU 数)，环境变量仍优先。"""
    monkeypatch.setattr(cpu_threads, "effective_cpu_count", lambda: cpus)
    monkeypatch.delenv("MINERU_INTRA_OP_NUM_THREADS", raising=False)
    opts = ort.SessionOptions()
    onnx.configure_ort_threads(opts)
    assert opts.intra_op_num_threads == expected

    monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "3")
    opts = ort.SessionOptions()
    onnx.configure_ort_threads(opts)
    assert opts.intra_op_num_threads == 3
