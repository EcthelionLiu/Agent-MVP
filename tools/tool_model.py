class ToolModel:
    def __init__(self):
        self._registry = {}
    def register(self, name, func):
        if name in self._registry:
            raise ValueError("duplicate")
        self._registry[name] = func
    def run(self, name, **kw):
        return self._registry[name](**kw)

TOOL_MODEL = ToolModel()

def calc_pressure_drop(q, L=10, D=0.5):
    return 0.02 * (q/10000)**2 * (L/D)

def diagnose_pipeline(fault_type: str, data: dict) -> dict:
    # 根据 fault_type 和 data（dp, soil_shift 等），返回可能原因
    return {"root_causes":["seal_leak","erosion"], "confidence":[0.7,0.3]}

def propose_fix(root_cause: str) -> str:
    # Stub: 针对某个 root_cause 提供解决方案
    if root_cause=="seal_leak":
        return "检查并更换阀门密封圈；切换到旁通管路进行维修。"
    return "清理内腔并进行侵蚀防护处理。"

TOOL_MODEL.register("calc_dp", calc_pressure_drop)
TOOL_MODEL.register("diagnose_pipeline", diagnose_pipeline)
TOOL_MODEL.register("propose_fix", propose_fix)
