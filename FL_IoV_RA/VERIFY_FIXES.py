#!/usr/bin/env python3
"""
严谨修正验证脚本
检查所有修正是否正确应用
"""

import sys
import os

def verify_config():
    """验证config.py修正"""
    print("\n" + "="*70)
    print("【配置文件验证】config.py")
    print("="*70)

    try:
        import config

        checks = {
            "LAMBDA1 = 0.1（能效权重应该0.1）": config.LAMBDA1 == 0.1,
            "LAMBDA2 = 0.9（延迟权重应该0.9）": config.LAMBDA2 == 0.9,
            "EPSILON_DECAY = 800（前800个Episode衰减）": config.EPSILON_DECAY == 800,
            "BUFFER_SIZE = 30000（3GPP规范）": config.BUFFER_SIZE == 30000,
            "LEARNING_RATE = 0.001（RMSProp学习率）": abs(config.LEARNING_RATE - 1e-3) < 1e-10,
            "TARGET_UPDATE_FREQ = 200（硬更新频率）": config.TARGET_UPDATE_FREQ == 200,
        }

        all_pass = True
        for check_name, result in checks.items():
            status = "✅ PASS" if result else "❌ FAIL"
            value = getattr(config, check_name.split('=')[0].strip().upper(), "N/A")
            print(f"  {check_name:50} {status}")
            if not result:
                all_pass = False

        return all_pass

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def verify_optimizer():
    """验证优化器修正"""
    print("\n" + "="*70)
    print("【优化器验证】agents/dqn_agent.py")
    print("="*70)

    try:
        # 检查源代码中是否有RMSProp
        with open("agents/dqn_agent.py", "r", encoding="utf-8") as f:
            code = f.read()

        checks = {
            "使用RMSProp而不是Adam": "optim.RMSprop" in code and "optim.Adam" not in code.split("self.optimizer")[1].split("\n")[0:3],
            "包含alpha=0.99参数": "alpha=0.99" in code,
            "包含eps=1e-8参数": "eps=1e-8" in code,
            "使用hard_update_dqn": "hard_update_dqn(self.target_q_network" in code,
        }

        all_pass = True
        for check_name, result in checks.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {check_name:50} {status}")
            if not result:
                all_pass = False

        return all_pass

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def verify_syntax():
    """验证Python语法"""
    print("\n" + "="*70)
    print("【语法验证】Python文件编译检查")
    print("="*70)

    files_to_check = [
        "config.py",
        "agents/dqn_agent.py",
        "env/iov_env.py",
        "networks/dqn_network.py",
    ]

    all_pass = True
    for filepath in files_to_check:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, filepath, "exec")
            print(f"  {filepath:40} ✅ PASS")
        except SyntaxError as e:
            print(f"  {filepath:40} ❌ FAIL: {e}")
            all_pass = False
        except Exception as e:
            print(f"  {filepath:40} ⚠️  {e}")

    return all_pass


def main():
    """主验证函数"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🔍 严谨修正验证脚本" + " "*32 + "║")
    print("║" + " "*68 + "║")
    print("║ 检查所有修正是否正确应用到代码" + " "*33 + "║")
    print("╚" + "="*68 + "╝")

    # 运行所有验证
    config_ok = verify_config()
    optimizer_ok = verify_optimizer()
    syntax_ok = verify_syntax()

    # 总体结果
    print("\n" + "="*70)
    print("【总体结果】")
    print("="*70)

    results = {
        "配置文件": config_ok,
        "优化器": optimizer_ok,
        "语法检查": syntax_ok,
    }

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20} {status}")

    all_pass = all(results.values())

    print("\n" + "="*70)
    if all_pass:
        print("✅ 所有修正验证通过！")
        print("\n可以开始测试运行:")
        print("  python main.py --mode train --episodes 10 --seed 42")
        print("\n预期观察:")
        print("  ✓ Reward逐步增加（能效优先的效果）")
        print("  ✓ Energy逐步减少（λ₂=0.9的作用）")
        print("  ✓ Delay逐步减少")
        print("  ✓ Constraint Satisfaction Rate增加")
        return 0
    else:
        print("❌ 还有修正未完成或有错误！")
        print("\n请检查上面标记为 FAIL 的项目")
        return 1


if __name__ == "__main__":
    sys.exit(main())
