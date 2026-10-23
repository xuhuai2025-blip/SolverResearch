"""声明式 AFC/AVC 图。结构固定，数值与有效性只属于一次求解上下文。"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


# 维护数值状态、依赖或计算接口。
def indices(*values):
    arrays = [np.asarray(v, dtype=int).ravel() for v in values if np.size(v)]
    return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=int)


# 维护数值状态、依赖或计算接口。
@dataclass(frozen=True)
class EvaluationPlan:
    """只读编号与影响关系；不保存当前状态、缓存或试探。"""
    values: tuple
    affectedValues: tuple
    affectedRows: tuple
    rowProviders: tuple
    unknownCount: int

    # 维护数值状态、依赖或计算接口。
    def pattern(self):
        counts = [len(rows) for rows in self.affectedRows]
        return sparse.csc_matrix((np.ones(sum(counts), bool),
            np.concatenate(self.affectedRows) if sum(counts) else np.empty(0, int),
            np.r_[0, np.cumsum(counts)]), shape=(self.unknownCount, self.unknownCount))
