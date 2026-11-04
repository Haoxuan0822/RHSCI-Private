"""仅供诊断：在当前 WBC 上恢复旧权重使用的导航语义。"""

import ast
import hashlib
import inspect
import textwrap

# 原导航：enterx-x/Psi0@bfffc460/scripts/eval_wbc_navigation.py。
# 历史v3速度选择器：@1f96adda/scripts/rsci_between_settings.py:navigation_method。
# 三处差异已与当前原生方法逐行核对；不复制整个下身控制器或 ONNX。
SOURCE_SHA256 = "152a19192c120bfb31c576b13ea74221b27febc4d7db088dbfec826b56f8754e"


def install():
    """校验原生方法后恢复旧门限/站走选择/返回命令，返回实际来源信息。"""
    from decoupled_wbc.control.policy import g1_gear_wbc_policy as module

    cls = module.G1GearWbcPolicy
    source = textwrap.dedent(inspect.getsource(cls.get_action))
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Native WBC method changed; refuse historical navigation adapter")
    gate = ast.dump(
        ast.parse(
            "abs(yaw_error) < YAW_ERROR_THRESHOLD or np.abs(vyaw_flag) < 0.1", mode="eval"
        ).body
    )
    switch = ast.dump(
        ast.parse(
            'os.environ.get("MODIFIED_POLICY_SWITCH_BEHAVIOR", "0").lower() in ("1", "true")',
            mode="eval",
        ).body
    )
    changes = [0, 0, 0]

    class Restore(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node.test) == gate:
                node.test = ast.parse("abs(yaw_error) < YAW_ERROR_THRESHOLD", mode="eval").body
                changes[0] += 1
            elif ast.dump(node.test) == switch:
                changes[1] += 1
                # manifest对应v3仅看前三维速度；旧v2四维范数不是本次历史协议。
                return ast.parse(
                    "switch_cond = np.linalg.norm(synced_navigate_cmd[:3]) < 0.05"
                ).body[0]
            return self.generic_visit(node)

        def visit_Dict(self, node):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "navigate_cmd":
                    node.values[index] = ast.Name(id="synced_navigate_cmd", ctx=ast.Load())
                    changes[2] += 1
            return self.generic_visit(node)

    tree = Restore().visit(ast.parse(source))
    if changes != [1, 1, 1]:
        raise RuntimeError(f"Unexpected historical navigation changes: {changes}")
    namespace = dict(vars(module))
    exec(
        compile(ast.fix_missing_locations(tree), "<diagnostic-legacy-navigation>", "exec"),
        namespace,
    )
    cls.get_action = namespace["get_action"]
    return {
        "protocol": "legacy-rate-v1-speed-selector-v1",
        "selector": "norm([vx,vy,computed_yaw_rate]) < 0.05",
        "native_method_sha256": SOURCE_SHA256,
        "adapter_function_sha256": hashlib.sha256(inspect.getsource(install).encode()).hexdigest(),
    }
