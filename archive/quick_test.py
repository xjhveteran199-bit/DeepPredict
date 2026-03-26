# -*- coding: utf-8 -*-
"""改进的测试：强制等待训练完成并截图"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.makedirs(r'C:/Users/XJH/DeepPredict/test_screenshots', exist_ok=True)

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    print("1. 打开页面...")
    page.goto('http://127.0.0.1:7860/')
    page.wait_for_timeout(3000)
    
    print("2. 切换到工具使用标签...")
    page.evaluate("""
        var tabs = document.querySelectorAll('[role="tab"]');
        for (var i = 0; i < tabs.length; i++) {
            if (tabs[i].textContent.includes('工具使用')) {
                tabs[i].click(); break;
            }
        }
    """)
    page.wait_for_timeout(3000)
    
    print("3. 上传文件...")
    inputs = page.query_selector_all('input[type="file"]')
    if inputs:
        inputs[0].set_input_files(r'C:\Users\XJH\Desktop\Raw_Data.csv')
        print("文件上传成功")
    page.wait_for_timeout(8000)
    
    # 检查数据预览是否正确加载
    body = page.text_content('body')
    if '2401' in body or 'K-with' in body:
        print("数据加载正确!")
    else:
        print("警告: 数据未正确加载")
    
    page.screenshot(path=r'C:/Users/XJH/DeepPredict/test_screenshots/01_uploaded.png', full_page=True)
    print("截图: 01_uploaded.png")
    
    # 找所有下拉菜单
    dropdowns = page.query_selector_all('[role="combobox"]')
    print(f"找到 {len(dropdowns)} 个 combobox")
    for dd in dropdowns:
        parent = dd.evaluate("el => el.parentElement ? el.parentElement.innerText.slice(0,100) : ''")
        print(f"  parent: {parent}")
    
    # 通过 label 找目标列下拉菜单
    print("\n4. 选择目标列 K-with epifluidics...")
    # 找包含"目标列"或"Y"的 label，然后找对应的 combobox
    labels = page.query_selector_all('label, span, p')
    target_label = None
    for lbl in labels:
        txt = lbl.text_content().strip()
        if '目标列' in txt or ('Y' in txt and '选择' in txt):
            target_label = lbl
            break
    
    if target_label:
        # 找相邻的 combobox
        print(f"找到目标列标签: {target_label.text_content()[:50]}")
    
    # 直接通过点击下拉菜单并选择选项
    all_dropdowns = page.query_selector_all('[role="combobox"]')
    if len(all_dropdowns) >= 2:
        # 点击目标列下拉菜单
        all_dropdowns[1].click()
        page.wait_for_timeout(800)
        
        # 等待 listbox 出现
        page.wait_for_selector('[role="listbox"]', timeout=3000)
        
        # 选择 K-with 选项
        opts = page.query_selector_all('[role="listbox"] option')
        for opt in opts:
            txt = opt.text_content().strip()
            print(f"  选项: {txt}")
            if 'K-with' in txt:
                opt.click()
                print(f"  已选择: {txt}")
                break
    page.wait_for_timeout(500)
    
    # 选择模型（CNN1D，不需要 Enhanced 版本，训练更快）
    print("\n5. 选择 CNN1D 模型...")
    all_dd = page.query_selector_all('[role="combobox"]')
    if len(all_dd) >= 3:
        all_dd[2].click()
        page.wait_for_timeout(800)
        
        page.wait_for_selector('[role="listbox"]', timeout=3000)
        opts = page.query_selector_all('[role="listbox"] option')
        for opt in opts:
            txt = opt.text_content().strip()
            if 'CNN1D' in txt and 'Enhanced' not in txt:
                opt.click()
                print(f"  已选择: {txt}")
                break
    page.wait_for_timeout(500)
    
    # 设置预测时长
    print("\n6. 设置预测时长...")
    num_inputs = page.query_selector_all('input[type="number"]')
    for inp in num_inputs:
        aria = inp.get_attribute('aria-label') or ''
        ph = inp.get_attribute('placeholder') or ''
        if '预测' in aria or '分钟' in aria or ph:
            inp.fill('10')
            print(f"  已填: 10分钟")
            break
    
    page.screenshot(path=r'C:/Users/XJH/DeepPredict/test_screenshots/02_configured.png', full_page=True)
    print("截图: 02_configured.png")
    
    # 点击开始训练
    print("\n7. 开始训练...")
    train_btn = None
    for btn in page.query_selector_all('button'):
        if '开始训练' in btn.text_content():
            train_btn = btn
            break
    
    if train_btn:
        train_btn.click()
        print("训练开始!")
    else:
        print("错误: 找不到开始训练按钮")
        browser.close()
        sys.exit(1)
    
    # 等待训练完成（最长6分钟，每30秒检查一次）
    last_status = ""
    for elapsed in range(0, 360, 15):
        page.wait_for_timeout(15)
        body = page.text_content('body')
        
        # 提取进度信息
        import re
        progress_matches = re.findall(r'[\d.]+[sс]', body)
        status = progress_matches[-1] if progress_matches else f"{elapsed}s"
        
        if status != last_status:
            print(f"  [{elapsed}s] 状态: {status}")
            last_status = status
        
        # 检查关键结果
        if 'error' in body.lower() or '错误' in body or '失败' in body or 'Permission' in body:
            print(f"[{elapsed}s] 检测到错误!")
            break
        
        # 检查是否完成（不再显示百分比进度）
        if elapsed > 30 and '30%' not in body and '0.' not in body.split('未来趋势预测')[1][:200] if '未来趋势预测' in body else False:
            print(f"[{elapsed}s] 可能完成!")
        
        # 检查训练结果文本（包含R²数值）
        if 'R' in body and ('²' in body or '2' in body) and '.' in body:
            # 更精确的检查
            r2_match = re.search(r'R[²2][\s:=]+[\d.]+', body)
            if r2_match:
                print(f"[{elapsed}s] SUCCESS! 找到: {r2_match.group()}")
                break
        
        if elapsed >= 350:
            print(f"[{elapsed}s] 超时!")
            break
    
    # 最终截图
    page.screenshot(path=r'C:/Users/XJH/DeepPredict/test_screenshots/03_result.png', full_page=True)
    print("\n最终截图已保存")
    
    # 输出关键文本
    body = page.text_content('body')
    if '未来趋势预测' in body:
        idx = body.index('未来趋势预测')
        section = body[idx:idx+500]
        print("\n=== 预测结果区域 ===")
        print(section[:500])
    
    browser.close()
    print("\n测试完成!")
