"""本地快速接话层(Local Backchannel Layer)。

在主 LLM 返回前显示一句很短的角色化过渡反应(字幕 + 表情 + 可选预合成语音)。

模块划分:
- models:词表常量与数据类(标签、模板、变体、清单)
- manifest:角色包 backchannels manifest 的加载与校验
- classifier:规则分类器(用户意图 + 情绪)
- resolver:模板匹配(相位 > 精确 > 同意图 > 兜底)与防重复轮换
"""
