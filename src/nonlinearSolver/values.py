"""求解器内部的轻量记录容器，不依赖应用或领域模型。"""


# 维护数值状态、依赖或计算接口。
class Struct(dict):
    """以键或属性访问同一份求解状态与统计。"""
    # 维护数值状态、依赖或计算接口。
    def __getattr__(self,key):
        try:return self[key]
        except KeyError as error:raise AttributeError(key) from error
    # 维护数值状态、依赖或计算接口。
    def __setattr__(self,key,value):self[key]=value
    # 维护数值状态、依赖或计算接口。
    def copy(self):return Struct(self)
