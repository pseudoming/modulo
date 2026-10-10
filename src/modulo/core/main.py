"""日志导入主流程：读取和提取 → 筛选 → 复算 → 汇总 → 保存。"""

from collections.abc import Callable

from .domain.data_models import BatchResult
from .domain.interfaces import Calculator, ProtocolAdapter, RawArchive, SourceAdapter
from .domain.platforms import PlatformConfig
from .processing.aggregation import summarize
from .processing.ingestion import collect_calls
from .processing.recalculation import calculate_calls
from .processing.selection import select_calls, selection_summary


def run[SavedResult](
    source: SourceAdapter,
    protocol: ProtocolAdapter,
    calculator: Calculator,
    *,
    archive: RawArchive,
    save: Callable[[BatchResult], SavedResult],
    namespace: str,
    payload_field: str | None = None,
    max_records: int = 10_000,
    platform_config: PlatformConfig | None = None,
) -> SavedResult:
    """接收已经配置好的适配器与保存函数，按顺序执行完整导入。"""
    if not namespace or max_records < 1:
        raise ValueError("namespace and a positive max_records are required")
    platform_config = platform_config or PlatformConfig()

    # 1. 读取源日志、归档原文，按协议提取调用并处理关联、重复和冲突。
    batch = collect_calls(
        source,
        protocol,
        archive,
        namespace=namespace,
        payload_field=payload_field,
        max_records=max_records,
        platform_config=platform_config,
        bindings=calculator.bindings,
    )

    # 2. 按平台与模型筛选；未入选记录保留原文和诊断。
    selection = selection_summary(batch, platform_config)
    selected = select_calls(batch, platform_config, selection)

    # 3. 复算入选调用，保留分项状态、差额与失败原因。
    rows = calculate_calls(selected, calculator, batch.diagnostics)

    # 4. 汇总本批次的结果与实际读取范围。
    result = summarize(
        rows,
        batch.counters,
        batch.diagnostics,
        selection,
        batch_read_complete=batch.complete,
    )

    # 5. 保存完整结果；保存失败直接向上传递，不能返回成功。
    return save(result)
