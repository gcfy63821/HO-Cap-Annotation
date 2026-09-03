# 志愿者标注质量详细报告

**统计日期**：2026-07-26  **总条数**：4058  

## 分类说明

| 分类 | 严重程度 | 说明 |
|------|---------|------|
| 全无操作对象标注 | 🔴 最严重 | 所有相机的操作对象角色均无标注点 |
| 所有标注仅第0帧 | 🔴 严重 | 主工具和操作对象只有第0帧有标注 |
| 部分相机缺操作对象 | 🟠 较严重 | 某些相机视角操作对象无标注 |
| 操作对象缺第0帧 | 🟠 较严重 | 操作对象标注缺少起始帧 |
| 操作对象仅第0帧 | 🟠 较严重 | 操作对象只有第0帧，无法覆盖后段 |
| 主工具缺第0帧 | 🟡 一般 | 主工具缺起始帧 |
| 主工具仅第0帧 | 🟡 一般 | 主工具只标了第0帧 |
| 操作对象缺100%帧 | 🟡 一般 | 操作对象缺末尾帧 |
| 主工具缺100%帧 | 🟡 一般 | 主工具缺末尾帧 |
| 仅缺50%帧(旧UI) | ✅ 可接受 | 旧版界面显示20%帧而非50%帧，数据仍可用 |
| 完整 | ✅ | 所有关键帧均已标注 |

---

## 全局汇总

| 分类 | 条数 | 占比 |
|------|------|------|
| 🔴 全无操作对象标注 | 238 | 5.9% |
| 🔴 所有标注仅第0帧 | 458 | 11.3% |
| 🟠 部分相机缺操作对象 | 686 | 16.9% |
| 🟠 操作对象缺第0帧 | 266 | 6.6% |
| 🟠 操作对象仅第0帧 | 484 | 11.9% |
| 🟡 主工具缺第0帧 | 413 | 10.2% |
| 🟡 主工具仅第0帧 | 27 | 0.7% |
| 🟡 操作对象缺100%帧 | 591 | 14.6% |
| 🟡 主工具缺100%帧 | 95 | 2.3% |
| ✅ 仅缺50%帧(旧UI) | 303 | 7.5% |
| ✅ 完整 | 496 | 12.2% |
| 🔴 无prompt文件 | 1 | 0.0% |

---

## 各标注者详细情况

### zhangqihui#13939771539
**总条数**：654 条 ｜ **完整**：109 条 ｜ **可接受(含旧UI)**：159 条

**分类统计：**

- 🔴 **全无操作对象标注**：4 条
- 🟠 **部分相机缺操作对象**：47 条
- 🟠 **操作对象缺第0帧**：18 条
- 🟠 **操作对象仅第0帧**：182 条
- 🟡 **主工具缺第0帧**：85 条
- 🟡 **主工具仅第0帧**：4 条
- 🟡 **操作对象缺100%帧**：148 条
- 🟡 **主工具缺100%帧**：7 条
- ✅ **仅缺50%帧(旧UI)**：50 条

<details>
<summary>🔴 全无操作对象标注（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1748 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_45 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1749 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_46 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2059 | 20260115_pinkknife_slice_unpealed_banana_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam7_rgb |
| 2087 | 20260118_orangeknife_slice_peeled_banana_18 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（47 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 181 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_52 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam3_rgb …共5项 |
| 182 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_53 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 473 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_75 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 474 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_76 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 485 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_86 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 487 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 584 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_12 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 586 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_14 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam7_rgb …共5项 |
| 589 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 592 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_2 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb …共5项 |
| 595 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam7_rgb …共4项 |
| 602 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 2007 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2407 | 20260119_greenstraw_stir_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 2447 | 20260119_yellowstraw_stir_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2448 | 20260119_yellowstraw_stir_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2452 | 20260119_yellowstraw_stir_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2475 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2789 | 20260121_smallwoodenspoon_crush_pealed_banana_9 | 操作对象无标注: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3373 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3375 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3382 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3386 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3390 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3391 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3394 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3401 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3402 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3404 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3405 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3406 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3410 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3415 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3417 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3418 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3419 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3421 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_47 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共4项 |
| 3453 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_79 | 操作对象无标注: cam1_rgb; 主工具缺末帧: cam5_rgb |
| 4016 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam2_rgb |
| 4017 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4018 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4019 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4024 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4025 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4026 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_19 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb |
| 4027 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4029 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（18 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 577 | 20260105_smallplate_greenchopstick_slice_smalldough_6 | 操作对象缺第0帧: cam2_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 590 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_18 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb …共4项 |
| 591 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_19 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam0_rgb …共4项 |
| 598 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_25 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 2021 | 20260115_orangeknife_slice_unpealed_banana_21 | 操作对象缺第0帧: cam7_rgb |
| 2024 | 20260115_orangeknife_slice_unpealed_banana_24 | 操作对象缺第0帧: cam7_rgb |
| 2026 | 20260115_orangeknife_slice_unpealed_banana_26 | 操作对象缺第0帧: cam7_rgb |
| 2068 | 20260115_pinkknife_slice_unpealed_banana_27 | 操作对象缺第0帧: cam7_rgb |
| 2072 | 20260115_pinkknife_slice_unpealed_banana_4 | 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2073 | 20260115_pinkknife_slice_unpealed_banana_5 | 操作对象缺第0帧: cam7_rgb |
| 2085 | 20260118_orangeknife_slice_peeled_banana_16 | 操作对象缺第0帧: cam3_rgb; 主工具缺第0帧: cam0_rgb |
| 2616 | 20260121_mallet_crush_pealed_banana_61 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb; 操作对象缺50%帧: cam1_rgb,cam3_rgb,cam6_rgb |
| 2787 | 20260121_smallwoodenspoon_crush_pealed_banana_7 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3426 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_52 | 操作对象仅第0帧: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 3433 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_59 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 3435 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_61 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb |
| 3437 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_63 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb |
| 3547 | 20260129_redcup_roll_largedough_on_plasticcutter_18 | 操作对象缺第0帧: cam1_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（182 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 527 | 20260104_smallplate_woodenchopstick_slice_smalldough_45 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 2604 | 20260121_mallet_crush_pealed_banana_50 | 操作对象仅第0帧: cam7_rgb |
| 2788 | 20260121_smallwoodenspoon_crush_pealed_banana_8 | 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam7_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3076 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_18 | 操作对象仅第0帧: cam0_rgb; 操作对象缺50%帧: cam1_rgb,cam4_rgb,cam7_rgb; 主工具缺50%帧: cam7_rgb |
| 3428 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_54 | 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3556 | 20260129_redcup_roll_largedough_on_plasticcutter_26 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam2_rgb |
| 3558 | 20260129_redcup_roll_largedough_on_plasticcutter_28 | 操作对象仅第0帧: cam2_rgb,cam6_rgb |
| 3569 | 20260129_redcup_roll_largedough_on_plasticcutter_38 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3570 | 20260129_redcup_roll_largedough_on_plasticcutter_39 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3571 | 20260129_redcup_roll_largedough_on_plasticcutter_4 | 操作对象仅第0帧: cam2_rgb |
| 3573 | 20260129_redcup_roll_largedough_on_plasticcutter_41 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具缺第0帧: cam6_rgb |
| 3576 | 20260129_redcup_roll_largedough_on_plasticcutter_44 | 操作对象仅第0帧: cam2_rgb |
| 3577 | 20260129_redcup_roll_largedough_on_plasticcutter_5 | 操作对象仅第0帧: cam2_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam6_rgb |
| 3581 | 20260129_redcup_roll_largedough_on_plasticcutter_9 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam6_rgb |
| 3582 | 20260129_whitecup_roll_dough_on_plasticcutter_1 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam6_rgb |
| 3583 | 20260129_whitecup_roll_dough_on_plasticcutter_10 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3610 | 20260129_whitecup_roll_dough_on_plasticcutter_35 | 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb |
| 3615 | 20260129_whitecup_roll_dough_on_plasticcutter_4 | 操作对象仅第0帧: cam2_rgb,cam3_rgb |
| 3616 | 20260129_whitecup_roll_dough_on_plasticcutter_40 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam6_rgb |
| 3617 | 20260129_whitecup_roll_dough_on_plasticcutter_41 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3618 | 20260129_whitecup_roll_dough_on_plasticcutter_42 | 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3619 | 20260129_whitecup_roll_dough_on_plasticcutter_43 | 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3620 | 20260129_whitecup_roll_dough_on_plasticcutter_44 | 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3624 | 20260129_whitecup_roll_dough_on_plasticcutter_48 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3625 | 20260129_whitecup_roll_dough_on_plasticcutter_49 | 操作对象仅第0帧: cam2_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3626 | 20260129_whitecup_roll_dough_on_plasticcutter_5 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb |
| 3627 | 20260129_whitecup_roll_dough_on_plasticcutter_50 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3628 | 20260129_whitecup_roll_dough_on_plasticcutter_51 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3629 | 20260129_whitecup_roll_dough_on_plasticcutter_6 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb; 主工具仅第0帧: cam6_rgb |
| 3630 | 20260129_whitecup_roll_dough_on_plasticcutter_7 | 操作对象仅第0帧: cam2_rgb |
| 3631 | 20260129_whitecup_roll_dough_on_plasticcutter_8 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3632 | 20260129_whitecup_roll_dough_on_plasticcutter_9 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb |
| 3633 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_1 | 操作对象仅第0帧: cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3634 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_10 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3635 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_11 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3637 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_13 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3638 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_14 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3639 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_15 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3640 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_16 | 操作对象仅第0帧: cam0_rgb,cam4_rgb |
| 3641 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_17 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3644 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_2 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb |
| 3645 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_20 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam4_rgb; 主工具仅第0帧: cam0_rgb |
| 3646 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_21 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3647 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_22 | 操作对象仅第0帧: cam7_rgb |
| 3648 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_23 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3649 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_24 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3650 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_25 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3651 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_26 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3652 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_27 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3653 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_28 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3654 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_29 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3655 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_3 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3656 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_30 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3657 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_31 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3658 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_32 | 操作对象仅第0帧: cam0_rgb,cam7_rgb |
| 3659 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_33 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3660 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_34 | 操作对象仅第0帧: cam0_rgb,cam7_rgb |
| 3661 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_35 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3662 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_36 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3664 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_38 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3665 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_39 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3666 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_4 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3668 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_41 | 操作对象仅第0帧: cam0_rgb,cam7_rgb |
| 3669 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_42 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3670 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_43 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3671 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_5 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3672 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_6 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3673 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_7 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3674 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_8 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3675 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_9 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3676 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_1 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3677 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_10 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3678 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_11 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3680 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_13 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3681 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_14 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3682 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_15 | 操作对象仅第0帧: cam0_rgb,cam4_rgb |
| 3683 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_16 | 操作对象仅第0帧: cam0_rgb,cam4_rgb |
| 3685 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_18 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3686 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_19 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3687 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_2 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3688 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_20 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3689 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_21 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3690 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_22 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3691 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_23 | 操作对象仅第0帧: cam0_rgb,cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3692 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_24 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb |
| 3693 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_25 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3694 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam7_rgb |
| 3696 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_28 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3697 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_29 | 操作对象仅第0帧: cam0_rgb,cam7_rgb |
| 3698 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_3 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3701 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_32 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3702 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3703 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_34 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3704 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_35 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3705 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_36 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb |
| 3706 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_37 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3707 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_38 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3708 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_39 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3709 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_4 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb |
| 3710 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_40 | 操作对象仅第0帧: cam0_rgb,cam7_rgb |
| 3711 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_41 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3712 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_42 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3713 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_43 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3714 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_44 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3715 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_45 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3716 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_46 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3717 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_47 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3718 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_5 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3719 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_6 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3720 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_7 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3721 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_8 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3722 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_9 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3737 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_14 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3738 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_15 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3813 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_84 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3814 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_85 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3816 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_87 | 操作对象仅第0帧: cam7_rgb |
| 3817 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_88 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3818 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_89 | 操作对象仅第0帧: cam7_rgb |
| 3819 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_9 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3820 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_90 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3822 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_92 | 操作对象仅第0帧: cam7_rgb |
| 3823 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_93 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3824 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_94 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3825 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_95 | 操作对象仅第0帧: cam5_rgb,cam7_rgb |
| 3826 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_96 | 操作对象仅第0帧: cam1_rgb,cam7_rgb |
| 3827 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_97 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3828 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_98 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3840 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_19 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3843 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_21 | 操作对象仅第0帧: cam0_rgb |
| 3844 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_22 | 操作对象仅第0帧: cam0_rgb,cam4_rgb |
| 3845 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_23 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3846 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_24 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3847 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_25 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3848 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_26 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3849 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_27 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3850 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_28 | 操作对象仅第0帧: cam0_rgb,cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3851 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_29 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3852 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_3 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3853 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_30 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3854 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_31 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3856 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_33 | 操作对象仅第0帧: cam7_rgb; 主工具缺末帧: cam4_rgb,cam7_rgb |
| 3857 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_34 | 操作对象仅第0帧: cam7_rgb |
| 3858 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_35 | 操作对象仅第0帧: cam0_rgb |
| 3859 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_36 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3862 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_39 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3863 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_4 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3864 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_40 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3865 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_41 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam7_rgb |
| 3866 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_42 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 3868 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_44 | 操作对象仅第0帧: cam0_rgb |
| 3869 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_45 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb |
| 3871 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_47 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb,cam7_rgb |
| 3872 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_48 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3873 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_49 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3874 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_5 | 操作对象仅第0帧: cam4_rgb,cam7_rgb |
| 3875 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_50 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3876 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_51 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3877 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_52 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3878 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_53 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam4_rgb,cam7_rgb |
| 3879 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_54 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3880 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_55 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3881 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_56 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3887 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_61 | 操作对象仅第0帧: cam0_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3888 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_62 | 操作对象仅第0帧: cam4_rgb; 操作对象缺末帧: cam0_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3889 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_63 | 操作对象仅第0帧: cam0_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3890 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_64 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3891 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_7 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3892 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_8 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3893 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_9 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3911 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb; 主工具缺第0帧: cam7_rgb |
| 3913 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_20 | 操作对象仅第0帧: cam6_rgb |
| 3914 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_21 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3916 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_23 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3917 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_24 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam6_rgb |
| 3919 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb |
| 3921 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_28 | 操作对象仅第0帧: cam1_rgb,cam2_rgb; 主工具仅第0帧: cam0_rgb |
| 3923 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_3 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3924 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb |
| 3926 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_32 | 操作对象仅第0帧: cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb |
| 3927 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_33 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 4007 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_46 | 操作对象仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（85 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 74 | 20260105_largeplate_greenchopstick_slice_largedough_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 77 | 20260105_largeplate_greenchopstick_slice_largedough_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb,cam3_rgb,cam7_rgb …共4项 |
| 78 | 20260105_largeplate_greenchopstick_slice_largedough_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 79 | 20260105_largeplate_greenchopstick_slice_largedough_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 80 | 20260105_largeplate_greenchopstick_slice_largedough_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 81 | 20260105_largeplate_greenchopstick_slice_largedough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 84 | 20260105_largeplate_greenchopstick_slice_largedough_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 88 | 20260105_largeplate_greenchopstick_slice_largedough_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 94 | 20260105_largeplate_greenchopstick_slice_largedough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 97 | 20260105_largeplate_greenchopstick_slice_largedough_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 99 | 20260105_largeplate_greenchopstick_slice_largedough_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 101 | 20260105_largeplate_greenchopstick_slice_largedough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 492 | 20260104_smallplate_woodenchopstick_slice_smalldough_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 493 | 20260104_smallplate_woodenchopstick_slice_smalldough_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 510 | 20260104_smallplate_woodenchopstick_slice_smalldough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 548 | 20260105_smallplate_greenchopstick_slice_smalldough_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 556 | 20260105_smallplate_greenchopstick_slice_smalldough_24 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb |
| 563 | 20260105_smallplate_greenchopstick_slice_smalldough_30 | 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 568 | 20260105_smallplate_greenchopstick_slice_smalldough_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 571 | 20260105_smallplate_greenchopstick_slice_smalldough_38 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 572 | 20260105_smallplate_greenchopstick_slice_smalldough_39 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 574 | 20260105_smallplate_greenchopstick_slice_smalldough_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 575 | 20260105_smallplate_greenchopstick_slice_smalldough_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 975 | 20260108_squeegee_sweep_peanuts_nuts_from_table_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 981 | 20260108_squeegee_sweep_peanuts_nuts_from_table_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 982 | 20260108_squeegee_sweep_peanuts_nuts_from_table_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 984 | 20260108_squeegee_sweep_peanuts_nuts_from_table_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1381 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1655 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1690 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_49 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1692 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1693 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1694 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1695 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1696 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1697 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_55 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1699 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_57 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1700 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_58 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1701 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_59 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1703 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_60 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1704 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_61 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1705 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_62 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1729 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1751 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1754 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1755 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1756 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1757 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1758 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1759 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_55 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1760 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1761 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_57 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1762 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_58 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 2031 | 20260115_orangeknife_slice_unpealed_banana_30 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2049 | 20260115_pinkknife_slice_unpealed_banana_1 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2070 | 20260115_pinkknife_slice_unpealed_banana_29 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 2251 | 20260118_pinkknife_slice_peeled_banana_95 | 主工具缺第0帧: cam0_rgb |
| 2262 | 20260118_pinkknife_slice_unpealed_banana_40 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2790 | 20260122_squeegee_collect_sand_from_table_1 | 主工具缺第0帧: cam3_rgb |
| 3058 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_91 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3067 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_1 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3121 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_7 | 主工具缺第0帧: cam6_rgb |
| 3123 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_9 | 主工具缺第0帧: cam6_rgb |
| 3159 | 20260123_woodenspatula_collect_largeamount_sand_from_table_36 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3180 | 20260123_woodenspatula_collect_smallamount_sand_from_table_4 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3181 | 20260123_woodenspatula_collect_smallamount_sand_from_table_5 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3183 | 20260123_woodenspatula_collect_smallamount_sand_from_table_7 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3187 | 20260126_woodenspatula_collect_smallamount_sand_from_table_25 | 主工具缺第0帧: cam0_rgb |
| 3189 | 20260126_woodenspatula_collect_smallamount_sand_from_table_27 | 主工具缺第0帧: cam0_rgb |
| 3199 | 20260126_woodenspatula_collect_smallamount_sand_from_table_37 | 主工具缺第0帧: cam0_rgb |
| 3206 | 20260126_woodenspatula_collect_smallamount_sand_from_table_44 | 主工具缺第0帧: cam0_rgb |
| 3207 | 20260126_woodenspatula_collect_smallamount_sand_from_table_45 | 主工具缺第0帧: cam0_rgb |
| 3208 | 20260126_woodenspatula_collect_smallamount_sand_from_table_46 | 主工具缺第0帧: cam0_rgb |
| 3209 | 20260123_squeegee_collect_sand_from_table_1 | 主工具仅第0帧: cam2_rgb; 主工具缺第0帧: cam7_rgb |
| 3211 | 20260123_squeegee_collect_sand_from_table_11 | 主工具缺第0帧: cam0_rgb,cam4_rgb |
| 3212 | 20260123_squeegee_collect_sand_from_table_12 | 主工具缺第0帧: cam3_rgb,cam4_rgb |
| 3214 | 20260123_squeegee_collect_sand_from_table_14 | 主工具缺第0帧: cam3_rgb |
| 3228 | 20260123_squeegee_collect_sand_from_table_27 | 主工具缺第0帧: cam7_rgb |
| 3234 | 20260123_squeegee_collect_sand_from_table_5 | 主工具缺第0帧: cam2_rgb |
| 3237 | 20260123_squeegee_collect_sand_from_table_8 | 主工具缺第0帧: cam0_rgb |
| 3494 | 20260129_plasticcup_roll_dough_on_plasticcutter_38 | 主工具缺第0帧: cam6_rgb |
| 3495 | 20260129_plasticcup_roll_dough_on_plasticcutter_39 | 操作对象缺50%帧: cam6_rgb; 主工具缺第0帧: cam6_rgb |
| 3883 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_58 | 操作对象缺末帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3884 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_59 | 操作对象缺末帧: cam0_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3886 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_60 | 操作对象缺末帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 501 | 20260104_smallplate_woodenchopstick_slice_smalldough_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3222 | 20260123_squeegee_collect_sand_from_table_21 | 主工具仅第0帧: cam4_rgb |
| 3537 | 20260129_plasticcup_roll_largedough_on_plasticcutter_9 | 操作对象缺50%帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam2_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3663 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_37 | 主工具仅第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（148 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 91 | 20260105_largeplate_greenchopstick_slice_largedough_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 109 | 20260105_largeplate_woodenchopstick_slice_largedough_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 112 | 20260105_largeplate_woodenchopstick_slice_largedough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 115 | 20260105_largeplate_woodenchopstick_slice_largedough_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 498 | 20260104_smallplate_woodenchopstick_slice_smalldough_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 500 | 20260104_smallplate_woodenchopstick_slice_smalldough_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 502 | 20260104_smallplate_woodenchopstick_slice_smalldough_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 503 | 20260104_smallplate_woodenchopstick_slice_smalldough_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 505 | 20260104_smallplate_woodenchopstick_slice_smalldough_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 533 | 20260104_smallplate_woodenchopstick_slice_smalldough_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 541 | 20260105_smallplate_greenchopstick_slice_smalldough_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 573 | 20260105_smallplate_greenchopstick_slice_smalldough_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 578 | 20260105_smallplate_greenchopstick_slice_smalldough_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 579 | 20260105_smallplate_greenchopstick_slice_smalldough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 971 | 20260108_squeegee_sweep_peanuts_nuts_from_table_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 973 | 20260108_squeegee_sweep_peanuts_nuts_from_table_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 974 | 20260108_squeegee_sweep_peanuts_nuts_from_table_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 986 | 20260108_squeegee_sweep_peanuts_nuts_from_table_5 | 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 990 | 20260108_squeegee_sweep_peanuts_nuts_from_table_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 991 | 20260108_towel_sweep_almond_nuts_from_table_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 992 | 20260108_towel_sweep_almond_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 993 | 20260108_towel_sweep_almond_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 994 | 20260108_towel_sweep_almond_nuts_from_table_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 996 | 20260108_towel_sweep_almond_nuts_from_table_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1001 | 20260108_towel_sweep_almond_nuts_from_table_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1004 | 20260108_towel_sweep_almond_nuts_from_table_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1009 | 20260108_towel_sweep_almond_nuts_from_table_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1010 | 20260108_towel_sweep_almond_nuts_from_table_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1013 | 20260108_towel_sweep_almond_nuts_from_table_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1082 | 20260108_towel_sweep_peanuts_nuts_from_table_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1091 | 20260109_scrubbrush_sweep_almond_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1224 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1647 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1650 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1651 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1652 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1653 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1654 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1658 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1659 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1660 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1661 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1662 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1663 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1664 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1665 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1666 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1667 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1668 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1669 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1670 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1671 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1673 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1674 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1675 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1676 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1677 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1678 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_38 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1679 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1680 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1681 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1682 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1684 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1685 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1686 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_45 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1687 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_46 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1688 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1689 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1691 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1698 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1702 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1706 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1707 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1708 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1709 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1710 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1711 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1712 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1713 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1714 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1715 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1716 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1717 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1718 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1719 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1720 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1721 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1722 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1723 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1724 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1725 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1726 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1727 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1728 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1730 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam5_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1731 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1732 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1733 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1734 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1735 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1736 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1737 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1738 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1739 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1740 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_38 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1741 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1742 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1743 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1744 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1745 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_42 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1746 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1747 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1750 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1752 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_49 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1753 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1763 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_59 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1764 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1765 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_60 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1766 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1767 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1768 | 20260114_smallwoodenspoon_largeplate_cut_kineticsand_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2020 | 20260115_orangeknife_slice_unpealed_banana_20 | 操作对象缺末帧: cam7_rgb |
| 2033 | 20260115_orangeknife_slice_unpealed_banana_32 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam1_rgb |
| 2112 | 20260118_orangeknife_slice_peeled_banana_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2271 | 20260116_mallet_crush_pealed_banana_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2516 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2599 | 20260121_mallet_crush_pealed_banana_46 | 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam3_rgb |
| 2614 | 20260121_mallet_crush_pealed_banana_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2817 | 20260122_squeegee_collect_sand_from_table_34 | 操作对象缺末帧: cam7_rgb |
| 2843 | 20260123_brush_collect_largeamount_sand_from_table_12 | 操作对象缺末帧: cam7_rgb |
| 2846 | 20260123_brush_collect_largeamount_sand_from_table_15 | 操作对象缺末帧: cam7_rgb |
| 2892 | 20260123_brush_collect_smallamount_sand_from_table_13 | 操作对象缺末帧: cam7_rgb |
| 3154 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_5 | 操作对象缺末帧: cam7_rgb |
| 3158 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_9 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam7_rgb |
| 3161 | 20260123_woodenspatula_collect_largeamount_sand_from_table_38 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 3167 | 20260123_woodenspatula_collect_smallamount_sand_from_table_13 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb |
| 3168 | 20260123_woodenspatula_collect_smallamount_sand_from_table_14 | 操作对象缺末帧: cam7_rgb |
| 3170 | 20260123_woodenspatula_collect_smallamount_sand_from_table_16 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam6_rgb,cam7_rgb |
| 3173 | 20260123_woodenspatula_collect_smallamount_sand_from_table_19 | 操作对象缺末帧: cam7_rgb |
| 3184 | 20260123_woodenspatula_collect_smallamount_sand_from_table_8 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam7_rgb |
| 3194 | 20260126_woodenspatula_collect_smallamount_sand_from_table_32 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 3202 | 20260126_woodenspatula_collect_smallamount_sand_from_table_40 | 操作对象缺末帧: cam7_rgb |
| 3478 | 20260129_plasticcup_roll_dough_on_plasticcutter_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3483 | 20260129_plasticcup_roll_dough_on_plasticcutter_28 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3506 | 20260129_plasticcup_roll_dough_on_plasticcutter_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3519 | 20260129_plasticcup_roll_largedough_on_plasticcutter_20 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3741 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_18 | 操作对象缺末帧: cam5_rgb,cam7_rgb |
| 3885 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（7 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2064 | 20260115_pinkknife_slice_unpealed_banana_23 | 操作对象缺50%帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2106 | 20260118_orangeknife_slice_peeled_banana_35 | 主工具缺末帧: cam5_rgb |
| 2842 | 20260123_brush_collect_largeamount_sand_from_table_11 | 主工具缺末帧: cam7_rgb |
| 2844 | 20260123_brush_collect_largeamount_sand_from_table_13 | 主工具缺末帧: cam7_rgb |
| 2924 | 20260123_brush_collect_smallamount_sand_from_table_9 | 操作对象缺50%帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3493 | 20260129_plasticcup_roll_dough_on_plasticcutter_37 | 操作对象缺50%帧: cam3_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3532 | 20260129_plasticcup_roll_largedough_on_plasticcutter_4 | 主工具缺末帧: cam7_rgb |

</details>

---

### zhangkaige#15670811832
**总条数**：652 条 ｜ **完整**：140 条 ｜ **可接受(含旧UI)**：208 条

**分类统计：**

- 🔴 **全无操作对象标注**：6 条
- 🟠 **部分相机缺操作对象**：207 条
- 🟠 **操作对象缺第0帧**：79 条
- 🟠 **操作对象仅第0帧**：41 条
- 🟡 **主工具缺第0帧**：64 条
- 🟡 **主工具仅第0帧**：8 条
- 🟡 **操作对象缺100%帧**：24 条
- 🟡 **主工具缺100%帧**：14 条
- ✅ **仅缺50%帧(旧UI)**：68 条
- 🔴 **无prompt文件**：1 条

<details>
<summary>🔴 全无操作对象标注（6 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2854 | 20260123_brush_collect_largeamount_sand_from_table_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2855 | 20260123_brush_collect_largeamount_sand_from_table_23 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2856 | 20260123_brush_collect_largeamount_sand_from_table_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2857 | 20260123_brush_collect_largeamount_sand_from_table_25 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3348 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3448 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_74 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（207 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 399 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_96 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam6_rgb …共4项 |
| 1908 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam6_rgb,cam7_rgb |
| 1910 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1912 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_29 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1913 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb |
| 1914 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1915 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1916 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1919 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1921 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1923 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1924 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1925 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1926 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺50%帧: cam5_rgb |
| 1928 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1929 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1931 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1933 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1936 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1939 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 1940 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1942 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1944 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_31 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1946 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_33 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1949 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1950 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1953 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1957 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_12 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 1958 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_13 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺50%帧: cam7_rgb |
| 1961 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_16 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 1962 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_17 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1963 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_18 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 主工具缺第0帧: cam7_rgb |
| 1964 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1966 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam7_rgb |
| 1968 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_22 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 主工具缺末帧: cam4_rgb …共4项 |
| 1978 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1986 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1988 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1998 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1999 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2001 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb |
| 2003 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2004 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam1_rgb |
| 2391 | 20260119_greenstraw_stir_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2392 | 20260119_greenstraw_stir_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam6_rgb |
| 2393 | 20260119_greenstraw_stir_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam5_rgb,cam6_rgb; 操作对象缺第0帧: cam1_rgb …共5项 |
| 2394 | 20260119_greenstraw_stir_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2395 | 20260119_greenstraw_stir_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2396 | 20260119_greenstraw_stir_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2397 | 20260119_greenstraw_stir_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 2398 | 20260119_greenstraw_stir_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2399 | 20260119_greenstraw_stir_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2400 | 20260119_greenstraw_stir_coffee_shallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2401 | 20260119_greenstraw_stir_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2402 | 20260119_greenstraw_stir_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2403 | 20260119_greenstraw_stir_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam5_rgb |
| 2404 | 20260119_greenstraw_stir_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2405 | 20260119_greenstraw_stir_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2406 | 20260119_greenstraw_stir_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2412 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2413 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2414 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2415 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 2416 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2417 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_18 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共4项 |
| 2418 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2419 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2420 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2421 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2422 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb |
| 2423 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2424 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2425 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2426 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2427 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2428 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2431 | 20260119_yellowstraw_stir_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2432 | 20260119_yellowstraw_stir_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam6_rgb |
| 2433 | 20260119_yellowstraw_stir_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2434 | 20260119_yellowstraw_stir_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2435 | 20260119_yellowstraw_stir_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2436 | 20260119_yellowstraw_stir_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb |
| 2437 | 20260119_yellowstraw_stir_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2438 | 20260119_yellowstraw_stir_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2442 | 20260119_yellowstraw_stir_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2443 | 20260119_yellowstraw_stir_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2445 | 20260119_yellowstraw_stir_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2446 | 20260119_yellowstraw_stir_coffee_shallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2449 | 20260119_yellowstraw_stir_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2450 | 20260119_yellowstraw_stir_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2451 | 20260119_yellowstraw_stir_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2453 | 20260119_yellowstraw_stir_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb |
| 2455 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2456 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2457 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2459 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2460 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2462 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2463 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2464 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2465 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2466 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2467 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2468 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2469 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 2470 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb …共4项 |
| 2471 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam0_rgb,cam7_rgb |
| 2472 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2474 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2476 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb …共4项 |
| 2619 | 20260121_mallet_crush_pealed_banana_64 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2640 | 20260121_mallet_crush_almond_nuts_8 | 操作对象无标注: cam6_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam2_rgb; 主工具缺第0帧: cam6_rgb …共4项 |
| 2647 | 20260121_mallet_crush_peanuts_nuts_14 | 操作对象无标注: cam6_rgb,cam7_rgb |
| 2730 | 20260122_mallet_crush_cashew_nuts_5 | 操作对象无标注: cam0_rgb,cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2744 | 20260121_largewoodenspoon_crush_pealed_banana_18 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam2_rgb |
| 2746 | 20260121_largewoodenspoon_crush_pealed_banana_2 | 操作对象无标注: cam1_rgb,cam7_rgb |
| 2749 | 20260121_largewoodenspoon_crush_pealed_banana_22 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 2750 | 20260121_largewoodenspoon_crush_pealed_banana_3 | 操作对象无标注: cam0_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb,cam6_rgb |
| 2751 | 20260121_largewoodenspoon_crush_pealed_banana_4 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2754 | 20260121_largewoodenspoon_crush_pealed_banana_7 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam4_rgb |
| 2759 | 20260121_smallwoodenspoon_crush_pealed_banana_11 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2760 | 20260121_smallwoodenspoon_crush_pealed_banana_12 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2761 | 20260121_smallwoodenspoon_crush_pealed_banana_13 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb; 主工具缺末帧: cam1_rgb |
| 2762 | 20260121_smallwoodenspoon_crush_pealed_banana_14 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2763 | 20260121_smallwoodenspoon_crush_pealed_banana_15 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺50%帧: cam6_rgb |
| 2764 | 20260121_smallwoodenspoon_crush_pealed_banana_16 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2765 | 20260121_smallwoodenspoon_crush_pealed_banana_17 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2770 | 20260121_smallwoodenspoon_crush_pealed_banana_21 | 操作对象无标注: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam4_rgb |
| 2774 | 20260121_smallwoodenspoon_crush_pealed_banana_25 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺50%帧: cam4_rgb |
| 2776 | 20260121_smallwoodenspoon_crush_pealed_banana_27 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb |
| 2777 | 20260121_smallwoodenspoon_crush_pealed_banana_28 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2780 | 20260121_smallwoodenspoon_crush_pealed_banana_30 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb; 主工具缺第0帧: cam0_rgb |
| 2781 | 20260121_smallwoodenspoon_crush_pealed_banana_31 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2782 | 20260121_smallwoodenspoon_crush_pealed_banana_32 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb |
| 2783 | 20260121_smallwoodenspoon_crush_pealed_banana_33 | 操作对象无标注: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3322 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3324 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 3325 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam1_rgb |
| 3329 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam2_rgb |
| 3331 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共6项 |
| 3332 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb |
| 3334 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3335 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb …共4项 |
| 3337 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 3338 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_6 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam7_rgb |
| 3341 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3342 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共6项 |
| 3344 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共6项 |
| 3345 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3346 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 3347 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 3349 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3350 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3351 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb …共4项 |
| 3352 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺末帧: cam0_rgb,cam7_rgb …共4项 |
| 3353 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3354 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb |
| 3355 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具仅第0帧: cam2_rgb |
| 3356 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_23 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam7_rgb …共4项 |
| 3357 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam4_rgb |
| 3358 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3359 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_26 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3360 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3361 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb …共4项 |
| 3363 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb |
| 3364 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3365 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3367 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb |
| 3368 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3370 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb |
| 3371 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb |
| 3372 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3376 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3377 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3378 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3379 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3380 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3381 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3383 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3384 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3385 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3387 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_6 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb …共4项 |
| 3388 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb |
| 3389 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共5项 |
| 3392 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3393 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam4_rgb |
| 3395 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象仅第0帧: cam4_rgb,cam7_rgb |
| 3396 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3397 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3398 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam3_rgb |
| 3399 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3400 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb |
| 3403 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb |
| 3414 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_31 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3420 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 3439 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_65 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3441 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_67 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam1_rgb,cam3_rgb |
| 3442 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_68 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3445 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_71 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3449 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_75 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam7_rgb |
| 3450 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_76 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb |
| 3451 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_77 | 操作对象无标注: cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb |
| 3454 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_80 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3456 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_82 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3457 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_83 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3460 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_86 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam7_rgb …共4项 |
| 3461 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_87 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（79 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1918 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_8 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |
| 1955 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_10 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |
| 2625 | 20260121_mallet_crush_pealed_banana_9 | 操作对象缺第0帧: cam2_rgb |
| 2627 | 20260121_mallet_crush_almond_nuts_10 | 操作对象缺第0帧: cam7_rgb |
| 2629 | 20260121_mallet_crush_almond_nuts_12 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam3_rgb,cam4_rgb |
| 2631 | 20260121_mallet_crush_almond_nuts_14 | 操作对象缺第0帧: cam7_rgb |
| 2632 | 20260121_mallet_crush_almond_nuts_15 | 操作对象缺第0帧: cam7_rgb |
| 2641 | 20260121_mallet_crush_almond_nuts_9 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2644 | 20260121_mallet_crush_peanuts_nuts_11 | 操作对象缺第0帧: cam7_rgb |
| 2645 | 20260121_mallet_crush_peanuts_nuts_12 | 操作对象缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2646 | 20260121_mallet_crush_peanuts_nuts_13 | 操作对象缺第0帧: cam7_rgb |
| 2648 | 20260121_mallet_crush_peanuts_nuts_15 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2649 | 20260121_mallet_crush_peanuts_nuts_16 | 操作对象缺第0帧: cam7_rgb; 主工具缺末帧: cam5_rgb |
| 2651 | 20260121_mallet_crush_peanuts_nuts_18 | 操作对象缺第0帧: cam7_rgb |
| 2658 | 20260121_mallet_crush_peanuts_nuts_24 | 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam2_rgb |
| 2661 | 20260121_mallet_crush_peanuts_nuts_27 | 操作对象缺第0帧: cam0_rgb |
| 2667 | 20260121_mallet_crush_peanuts_nuts_32 | 操作对象缺第0帧: cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |
| 2674 | 20260121_mallet_crush_peanuts_nuts_39 | 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2678 | 20260121_mallet_crush_peanuts_nuts_42 | 操作对象缺第0帧: cam0_rgb |
| 2679 | 20260121_mallet_crush_peanuts_nuts_43 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2681 | 20260121_mallet_crush_peanuts_nuts_45 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2683 | 20260121_mallet_crush_peanuts_nuts_47 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2687 | 20260121_mallet_crush_peanuts_nuts_6 | 操作对象缺第0帧: cam7_rgb |
| 2688 | 20260121_mallet_crush_peanuts_nuts_7 | 操作对象缺第0帧: cam6_rgb,cam7_rgb |
| 2692 | 20260122_mallet_crush_almond_nuts_18 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2694 | 20260122_mallet_crush_almond_nuts_20 | 操作对象缺第0帧: cam2_rgb |
| 2698 | 20260122_mallet_crush_almond_nuts_24 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2699 | 20260122_mallet_crush_almond_nuts_25 | 操作对象缺第0帧: cam2_rgb |
| 2701 | 20260122_mallet_crush_almond_nuts_27 | 操作对象缺第0帧: cam2_rgb |
| 2702 | 20260122_mallet_crush_almond_nuts_28 | 操作对象缺第0帧: cam2_rgb |
| 2703 | 20260122_mallet_crush_almond_nuts_29 | 操作对象缺第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 2704 | 20260122_mallet_crush_almond_nuts_30 | 操作对象缺第0帧: cam2_rgb |
| 2705 | 20260122_mallet_crush_almond_nuts_31 | 操作对象缺第0帧: cam2_rgb |
| 2706 | 20260122_mallet_crush_almond_nuts_32 | 操作对象缺第0帧: cam2_rgb |
| 2707 | 20260122_mallet_crush_almond_nuts_33 | 操作对象缺第0帧: cam2_rgb |
| 2708 | 20260122_mallet_crush_almond_nuts_34 | 操作对象缺第0帧: cam1_rgb,cam2_rgb |
| 2709 | 20260122_mallet_crush_almond_nuts_35 | 操作对象缺第0帧: cam1_rgb,cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2710 | 20260122_mallet_crush_almond_nuts_36 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2711 | 20260122_mallet_crush_almond_nuts_37 | 操作对象缺第0帧: cam1_rgb,cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2712 | 20260122_mallet_crush_almond_nuts_38 | 操作对象缺第0帧: cam7_rgb |
| 2713 | 20260122_mallet_crush_almond_nuts_39 | 操作对象缺第0帧: cam5_rgb; 操作对象缺末帧: cam7_rgb |
| 2714 | 20260122_mallet_crush_almond_nuts_40 | 操作对象缺第0帧: cam5_rgb; 操作对象缺末帧: cam7_rgb |
| 2715 | 20260122_mallet_crush_almond_nuts_41 | 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam6_rgb |
| 2716 | 20260122_mallet_crush_almond_nuts_42 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam5_rgb; 操作对象缺末帧: cam7_rgb |
| 2717 | 20260122_mallet_crush_almond_nuts_43 | 操作对象缺第0帧: cam0_rgb,cam4_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam6_rgb |
| 2721 | 20260122_mallet_crush_almond_nuts_47 | 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2723 | 20260122_mallet_crush_almond_nuts_49 | 操作对象缺第0帧: cam2_rgb |
| 2728 | 20260122_mallet_crush_cashew_nuts_3 | 操作对象缺第0帧: cam2_rgb; 主工具缺第0帧: cam1_rgb,cam4_rgb |
| 2729 | 20260122_mallet_crush_cashew_nuts_4 | 操作对象缺第0帧: cam2_rgb |
| 2731 | 20260122_mallet_crush_cashew_nuts_6 | 操作对象缺第0帧: cam2_rgb |
| 2733 | 20260122_mallet_crush_cashew_nuts_8 | 操作对象缺第0帧: cam2_rgb; 主工具缺末帧: cam3_rgb |
| 2734 | 20260122_mallet_crush_cashew_nuts_9 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam6_rgb |
| 2747 | 20260121_largewoodenspoon_crush_pealed_banana_20 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam5_rgb,cam6_rgb |
| 2748 | 20260121_largewoodenspoon_crush_pealed_banana_21 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam6_rgb |
| 2752 | 20260121_largewoodenspoon_crush_pealed_banana_5 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb |
| 2753 | 20260121_largewoodenspoon_crush_pealed_banana_6 | 操作对象缺第0帧: cam1_rgb,cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2755 | 20260121_largewoodenspoon_crush_pealed_banana_8 | 操作对象缺第0帧: cam7_rgb |
| 2756 | 20260121_largewoodenspoon_crush_pealed_banana_9 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam6_rgb |
| 2758 | 20260121_smallwoodenspoon_crush_pealed_banana_10 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2768 | 20260121_smallwoodenspoon_crush_pealed_banana_2 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam4_rgb |
| 2769 | 20260121_smallwoodenspoon_crush_pealed_banana_20 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb …共4项 |
| 2771 | 20260121_smallwoodenspoon_crush_pealed_banana_22 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb; 主工具缺第0帧: cam0_rgb |
| 2772 | 20260121_smallwoodenspoon_crush_pealed_banana_23 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb |
| 2775 | 20260121_smallwoodenspoon_crush_pealed_banana_26 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam3_rgb; 主工具缺末帧: cam2_rgb |
| 2778 | 20260121_smallwoodenspoon_crush_pealed_banana_29 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb; 操作对象缺50%帧: cam1_rgb …共4项 |
| 2779 | 20260121_smallwoodenspoon_crush_pealed_banana_3 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb,cam4_rgb |
| 2784 | 20260121_smallwoodenspoon_crush_pealed_banana_4 | 操作对象缺第0帧: cam1_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2956 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_127 | 操作对象缺第0帧: cam1_rgb,cam2_rgb; 操作对象缺50%帧: cam0_rgb |
| 2975 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_16 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb |
| 2977 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_18 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb |
| 2978 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_19 | 操作对象缺第0帧: cam1_rgb,cam2_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2981 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_21 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb |
| 2982 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_22 | 操作对象缺第0帧: cam1_rgb |
| 2986 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_26 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam7_rgb |
| 2987 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_27 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2996 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_35 | 操作对象缺第0帧: cam2_rgb; 主工具缺末帧: cam3_rgb,cam6_rgb,cam7_rgb |
| 3243 | 20260123_squeegee_collect_smallamount_sand_from_table_13 | 操作对象缺第0帧: cam2_rgb |
| 3484 | 20260129_plasticcup_roll_dough_on_plasticcutter_29 | 操作对象缺第0帧: cam5_rgb; 操作对象缺50%帧: cam6_rgb |
| 3514 | 20260129_plasticcup_roll_largedough_on_plasticcutter_16 | 操作对象缺第0帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb; 主工具缺50%帧: cam2_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（41 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1904 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2773 | 20260121_smallwoodenspoon_crush_pealed_banana_24 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam4_rgb |
| 3073 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_15 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具缺50%帧: cam0_rgb,cam7_rgb |
| 3074 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_16 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam3_rgb |
| 3081 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_22 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb |
| 3085 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_26 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb |
| 3095 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_35 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺50%帧: cam7_rgb |
| 3099 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_39 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 3102 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_41 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb |
| 3104 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_43 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺50%帧: cam7_rgb |
| 3107 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_46 | 操作对象仅第0帧: cam7_rgb; 主工具缺50%帧: cam7_rgb |
| 3108 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_47 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺50%帧: cam7_rgb |
| 3109 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_48 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam4_rgb; 主工具缺50%帧: cam7_rgb |
| 3110 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_49 | 操作对象仅第0帧: cam7_rgb; 主工具缺50%帧: cam4_rgb,cam7_rgb |
| 3113 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_51 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具仅第0帧: cam7_rgb |
| 3114 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_52 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb; 主工具缺50%帧: cam7_rgb |
| 3115 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_53 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺50%帧: cam7_rgb |
| 3116 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_54 | 操作对象仅第0帧: cam0_rgb; 操作对象缺50%帧: cam4_rgb,cam7_rgb |
| 3117 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_55 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb |
| 3149 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_32 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3188 | 20260126_woodenspatula_collect_smallamount_sand_from_table_26 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3191 | 20260126_woodenspatula_collect_smallamount_sand_from_table_29 | 操作对象仅第0帧: cam2_rgb |
| 3195 | 20260126_woodenspatula_collect_smallamount_sand_from_table_33 | 操作对象仅第0帧: cam1_rgb; 主工具仅第0帧: cam5_rgb; 主工具缺第0帧: cam0_rgb |
| 3226 | 20260123_squeegee_collect_sand_from_table_25 | 操作对象仅第0帧: cam2_rgb |
| 3230 | 20260123_squeegee_collect_sand_from_table_29 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam5_rgb |
| 3238 | 20260123_squeegee_collect_sand_from_table_9 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb |
| 3245 | 20260123_squeegee_collect_smallamount_sand_from_table_15 | 操作对象仅第0帧: cam1_rgb |
| 3261 | 20260123_squeegee_collect_smallamount_sand_from_table_3 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 3289 | 20260123_squeegee_collect_smallamount_sand_from_table_55 | 操作对象仅第0帧: cam0_rgb |
| 3895 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_2 | 操作对象仅第0帧: cam1_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3897 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_4 | 操作对象仅第0帧: cam3_rgb |
| 3898 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_5 | 操作对象仅第0帧: cam3_rgb,cam5_rgb,cam6_rgb |
| 3899 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_6 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam7_rgb |
| 3901 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_1 | 操作对象仅第0帧: cam2_rgb |
| 3902 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_10 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |
| 3903 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_11 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |
| 3904 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_12 | 操作对象仅第0帧: cam3_rgb,cam6_rgb |
| 3905 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_13 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具缺第0帧: cam0_rgb |
| 3906 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_14 | 操作对象仅第0帧: cam1_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |
| 3907 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_15 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |
| 3910 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_18 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（64 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2132 | 20260118_orangeknife_slice_peeled_banana_59 | 主工具缺第0帧: cam7_rgb |
| 2134 | 20260118_orangeknife_slice_peeled_banana_60 | 主工具缺第0帧: cam7_rgb |
| 2639 | 20260121_mallet_crush_almond_nuts_7 | 主工具缺第0帧: cam6_rgb |
| 2662 | 20260121_mallet_crush_peanuts_nuts_28 | 主工具缺第0帧: cam5_rgb |
| 2666 | 20260121_mallet_crush_peanuts_nuts_31 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2669 | 20260121_mallet_crush_peanuts_nuts_34 | 操作对象缺末帧: cam0_rgb; 主工具仅第0帧: cam6_rgb; 主工具缺第0帧: cam7_rgb |
| 2672 | 20260121_mallet_crush_peanuts_nuts_37 | 主工具缺第0帧: cam7_rgb |
| 2718 | 20260122_mallet_crush_almond_nuts_44 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2719 | 20260122_mallet_crush_almond_nuts_45 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2720 | 20260122_mallet_crush_almond_nuts_46 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb; 主工具缺末帧: cam2_rgb,cam7_rgb |
| 2722 | 20260122_mallet_crush_almond_nuts_48 | 操作对象缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |
| 2801 | 20260122_squeegee_collect_sand_from_table_2 | 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2811 | 20260122_squeegee_collect_sand_from_table_29 | 主工具缺第0帧: cam1_rgb |
| 2823 | 20260122_squeegee_collect_sand_from_table_4 | 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb |
| 2827 | 20260122_squeegee_collect_sand_from_table_43 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb |
| 2834 | 20260122_squeegee_collect_sand_from_table_5 | 主工具缺第0帧: cam1_rgb,cam3_rgb |
| 2836 | 20260122_squeegee_collect_sand_from_table_6 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam3_rgb |
| 2837 | 20260122_squeegee_collect_sand_from_table_7 | 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2838 | 20260122_squeegee_collect_sand_from_table_8 | 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2849 | 20260123_brush_collect_largeamount_sand_from_table_18 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb …共4项 |
| 2858 | 20260123_brush_collect_largeamount_sand_from_table_26 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 2860 | 20260123_brush_collect_largeamount_sand_from_table_28 | 主工具缺第0帧: cam6_rgb |
| 2861 | 20260123_brush_collect_largeamount_sand_from_table_29 | 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 2968 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_138 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 2969 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_139 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2971 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_140 | 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 2972 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_141 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2995 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_34 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2998 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_37 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2999 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_38 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam5_rgb |
| 3000 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_39 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3002 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_40 | 操作对象缺50%帧: cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3003 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_41 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3006 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_44 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3008 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_46 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3014 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_51 | 主工具缺第0帧: cam5_rgb |
| 3066 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_99 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3080 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_21 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 3082 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_23 | 操作对象缺50%帧: cam1_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3097 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_37 | 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam3_rgb,cam7_rgb |
| 3100 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_4 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3111 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_5 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3120 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_6 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 3122 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_8 | 操作对象缺50%帧: cam1_rgb; 主工具仅第0帧: cam4_rgb; 主工具缺第0帧: cam6_rgb …共4项 |
| 3157 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_8 | 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam7_rgb |
| 3162 | 20260123_woodenspatula_collect_largeamount_sand_from_table_39 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3166 | 20260123_woodenspatula_collect_smallamount_sand_from_table_12 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam2_rgb |
| 3174 | 20260123_woodenspatula_collect_smallamount_sand_from_table_2 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam2_rgb,cam7_rgb |
| 3177 | 20260123_woodenspatula_collect_smallamount_sand_from_table_22 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3178 | 20260123_woodenspatula_collect_smallamount_sand_from_table_23 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3179 | 20260123_woodenspatula_collect_smallamount_sand_from_table_3 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3186 | 20260126_woodenspatula_collect_smallamount_sand_from_table_24 | 主工具缺第0帧: cam0_rgb |
| 3196 | 20260126_woodenspatula_collect_smallamount_sand_from_table_34 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3198 | 20260126_woodenspatula_collect_smallamount_sand_from_table_36 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3233 | 20260123_squeegee_collect_sand_from_table_4 | 主工具缺第0帧: cam2_rgb |
| 3278 | 20260123_squeegee_collect_smallamount_sand_from_table_45 | 主工具缺第0帧: cam2_rgb |
| 3288 | 20260123_squeegee_collect_smallamount_sand_from_table_54 | 主工具缺第0帧: cam0_rgb |
| 3316 | 20260123_squeegee_collect_smallamount_sand_from_table_8 | 主工具缺第0帧: cam1_rgb |
| 3504 | 20260129_plasticcup_roll_dough_on_plasticcutter_7 | 主工具缺第0帧: cam2_rgb; 主工具缺50%帧: cam6_rgb |
| 3507 | 20260129_plasticcup_roll_largedough_on_plasticcutter_1 | 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb; 主工具缺第0帧: cam3_rgb,cam4_rgb |
| 3525 | 20260129_plasticcup_roll_largedough_on_plasticcutter_26 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3530 | 20260129_plasticcup_roll_largedough_on_plasticcutter_30 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3896 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_3 | 主工具缺第0帧: cam0_rgb |
| 3908 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_16 | 主工具缺第0帧: cam1_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（8 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2785 | 20260121_smallwoodenspoon_crush_pealed_banana_5 | 主工具仅第0帧: cam0_rgb |
| 3112 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_50 | 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3227 | 20260123_squeegee_collect_sand_from_table_26 | 主工具仅第0帧: cam6_rgb |
| 3232 | 20260123_squeegee_collect_sand_from_table_30 | 主工具仅第0帧: cam2_rgb |
| 3250 | 20260123_squeegee_collect_smallamount_sand_from_table_2 | 主工具仅第0帧: cam1_rgb |
| 3255 | 20260123_squeegee_collect_smallamount_sand_from_table_24 | 主工具仅第0帧: cam2_rgb |
| 3257 | 20260123_squeegee_collect_smallamount_sand_from_table_26 | 主工具仅第0帧: cam2_rgb |
| 3296 | 20260123_squeegee_collect_smallamount_sand_from_table_61 | 主工具仅第0帧: cam5_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（24 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2680 | 20260121_mallet_crush_peanuts_nuts_44 | 操作对象缺末帧: cam7_rgb |
| 2693 | 20260122_mallet_crush_almond_nuts_19 | 操作对象缺末帧: cam7_rgb |
| 2696 | 20260122_mallet_crush_almond_nuts_22 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2805 | 20260122_squeegee_collect_sand_from_table_23 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam1_rgb,cam5_rgb |
| 2816 | 20260122_squeegee_collect_sand_from_table_33 | 操作对象缺末帧: cam7_rgb |
| 2819 | 20260122_squeegee_collect_sand_from_table_36 | 操作对象缺末帧: cam7_rgb |
| 2820 | 20260122_squeegee_collect_sand_from_table_37 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2824 | 20260122_squeegee_collect_sand_from_table_40 | 操作对象缺末帧: cam7_rgb |
| 2825 | 20260122_squeegee_collect_sand_from_table_41 | 操作对象缺末帧: cam7_rgb |
| 2826 | 20260122_squeegee_collect_sand_from_table_42 | 操作对象缺末帧: cam7_rgb |
| 2828 | 20260122_squeegee_collect_sand_from_table_44 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam6_rgb |
| 2829 | 20260122_squeegee_collect_sand_from_table_45 | 操作对象缺末帧: cam7_rgb |
| 2830 | 20260122_squeegee_collect_sand_from_table_46 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2831 | 20260122_squeegee_collect_sand_from_table_47 | 操作对象缺末帧: cam7_rgb |
| 2832 | 20260122_squeegee_collect_sand_from_table_48 | 操作对象缺末帧: cam7_rgb |
| 2833 | 20260122_squeegee_collect_sand_from_table_49 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |
| 2852 | 20260123_brush_collect_largeamount_sand_from_table_20 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 3093 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_33 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam5_rgb; 主工具缺50%帧: cam7_rgb |
| 3118 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_56 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam4_rgb; 主工具缺末帧: cam7_rgb |
| 3169 | 20260123_woodenspatula_collect_smallamount_sand_from_table_15 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam5_rgb |
| 3172 | 20260123_woodenspatula_collect_smallamount_sand_from_table_18 | 操作对象缺末帧: cam7_rgb |
| 3190 | 20260126_woodenspatula_collect_smallamount_sand_from_table_28 | 操作对象缺末帧: cam7_rgb |
| 3481 | 20260129_plasticcup_roll_dough_on_plasticcutter_26 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam6_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb |
| 3511 | 20260129_plasticcup_roll_largedough_on_plasticcutter_13 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb; 主工具缺50%帧: cam2_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（14 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2129 | 20260118_orangeknife_slice_peeled_banana_56 | 主工具缺末帧: cam2_rgb,cam3_rgb |
| 2638 | 20260121_mallet_crush_almond_nuts_6 | 主工具缺末帧: cam3_rgb |
| 2671 | 20260121_mallet_crush_peanuts_nuts_36 | 主工具缺末帧: cam7_rgb |
| 2676 | 20260121_mallet_crush_peanuts_nuts_40 | 主工具缺末帧: cam1_rgb |
| 2724 | 20260122_mallet_crush_almond_nuts_50 | 主工具缺末帧: cam2_rgb |
| 2812 | 20260122_squeegee_collect_sand_from_table_3 | 主工具缺末帧: cam3_rgb |
| 2851 | 20260123_brush_collect_largeamount_sand_from_table_2 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam1_rgb |
| 2945 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_117 | 主工具缺末帧: cam1_rgb |
| 3025 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_61 | 操作对象缺50%帧: cam1_rgb,cam2_rgb; 主工具缺末帧: cam4_rgb |
| 3090 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_30 | 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3092 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_32 | 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam3_rgb; 主工具缺50%帧: cam7_rgb |
| 3096 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_36 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3160 | 20260123_woodenspatula_collect_largeamount_sand_from_table_37 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb |
| 3523 | 20260129_plasticcup_roll_largedough_on_plasticcutter_24 | 操作对象缺50%帧: cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |

</details>

---

### Yuan yueer#18188455260
**总条数**：520 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：3 条

**分类统计：**

- 🔴 **全无操作对象标注**：182 条
- 🔴 **所有标注仅第0帧**：331 条
- 🟠 **操作对象缺第0帧**：1 条
- 🟡 **主工具缺第0帧**：1 条
- 🟡 **操作对象缺100%帧**：1 条
- 🟡 **主工具缺100%帧**：1 条
- ✅ **仅缺50%帧(旧UI)**：3 条

<details>
<summary>🔴 全无操作对象标注（182 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 628 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_52 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 630 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_54 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 631 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_55 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 632 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_56 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 633 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_57 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 634 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_58 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 635 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_59 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 636 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_6 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 637 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_60 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 639 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_62 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 640 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_63 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 641 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_64 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 642 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_65 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 643 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_66 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 644 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_67 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 646 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_69 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 647 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_7 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 648 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_70 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 649 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_71 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 650 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_72 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 652 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_74 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 653 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_75 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 655 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_77 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 656 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_78 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb |
| 657 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_79 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb |
| 658 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_8 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 659 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_9 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 661 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_10 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 662 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_11 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 663 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_12 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 664 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_13 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 665 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_14 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 666 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_15 | 操作对象无标注: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 667 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 668 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_17 | 操作对象无标注: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 669 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_18 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 670 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 673 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_21 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 674 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 675 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_23 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 676 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 677 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_25 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 678 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_26 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 679 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_27 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 680 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_28 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 681 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_29 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 682 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_3 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 683 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_30 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 684 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 685 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_32 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 686 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_33 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 687 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_34 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 688 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_35 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 689 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_36 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 690 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_37 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 691 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 692 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_39 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 693 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_4 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 694 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_40 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 695 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_41 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 696 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_42 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 697 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_43 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 698 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_44 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 699 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_45 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 700 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_46 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 701 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_47 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 702 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_48 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 703 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_49 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 704 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_5 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 705 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_50 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 706 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_51 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 707 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_52 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 708 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_53 | 操作对象无标注: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 709 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_54 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 710 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_55 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 711 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_56 | 操作对象无标注: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 712 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_57 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 713 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_58 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 714 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_59 | 操作对象无标注: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 715 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_6 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 716 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_60 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 717 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_61 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 718 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_62 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 719 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_63 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 720 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_64 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 721 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_65 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 722 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_66 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 723 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_67 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 724 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_68 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 725 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_69 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 726 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_7 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 728 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_9 | 操作对象无标注: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 731 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_11 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 732 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_12 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 733 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 734 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_14 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 735 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_15 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 737 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_17 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 739 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 740 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_2 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 741 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_20 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 742 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_21 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 743 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 744 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_23 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 745 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 748 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_27 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 749 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_28 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 750 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_29 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 751 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_3 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 752 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_30 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 753 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 756 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_34 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 757 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_35 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 760 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 761 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_39 | 操作对象无标注: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 762 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_4 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 763 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_40 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 764 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_41 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 766 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_43 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 767 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_44 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 768 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_45 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 769 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_46 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 770 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_47 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 771 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_48 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 773 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_5 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 774 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_50 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 775 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_51 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 776 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_52 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 779 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_55 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 780 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_56 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 781 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_57 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 782 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_58 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 783 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_59 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 784 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_6 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 785 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_60 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 786 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_61 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 787 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_62 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 789 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_64 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 791 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_66 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 792 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_67 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 793 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_68 | 操作对象无标注: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 794 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_69 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 795 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_7 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 796 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_70 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 797 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_71 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 798 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_72 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 801 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_75 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 802 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_76 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 803 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_8 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 805 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_1 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 806 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_10 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 807 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_11 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 808 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_12 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 809 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 810 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_14 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 811 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_15 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 812 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 813 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_17 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 816 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_2 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 817 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_20 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 818 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_21 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 819 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 820 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_23 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 821 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 822 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_25 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 823 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_26 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 824 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_27 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 825 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_28 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 826 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_29 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 827 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_3 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 828 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_30 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 830 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_32 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 832 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_34 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 833 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_35 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 834 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_36 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 835 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_37 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 836 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 838 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_4 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 839 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_40 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1118 | 20260109_scrubbrush_sweep_almond_nuts_from_table_36 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1123 | 20260109_scrubbrush_sweep_almond_nuts_from_table_40 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1299 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🔴 所有标注仅第0帧（331 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 582 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_10 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 588 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam6_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 597 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 600 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_27 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 601 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_28 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 604 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_30 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 605 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 606 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_32 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 608 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_34 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 610 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_36 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 611 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_37 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 614 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 615 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_40 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 616 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_41 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 617 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_42 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 619 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_44 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 620 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_45 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 622 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_47 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 623 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_48 | 操作对象无标注: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 625 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_5 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 627 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_51 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1018 | 20260108_towel_sweep_cashew_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1019 | 20260108_towel_sweep_cashew_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1020 | 20260108_towel_sweep_cashew_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1021 | 20260108_towel_sweep_cashew_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1022 | 20260108_towel_sweep_cashew_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1023 | 20260108_towel_sweep_cashew_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1024 | 20260108_towel_sweep_cashew_nuts_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1025 | 20260108_towel_sweep_cashew_nuts_from_table_19 | 操作对象无标注: cam5_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1026 | 20260108_towel_sweep_cashew_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1027 | 20260108_towel_sweep_cashew_nuts_from_table_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1028 | 20260108_towel_sweep_cashew_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1029 | 20260108_towel_sweep_cashew_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1030 | 20260108_towel_sweep_cashew_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1031 | 20260108_towel_sweep_cashew_nuts_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1032 | 20260108_towel_sweep_cashew_nuts_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1033 | 20260108_towel_sweep_cashew_nuts_from_table_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1034 | 20260108_towel_sweep_cashew_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1035 | 20260108_towel_sweep_cashew_nuts_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1036 | 20260108_towel_sweep_cashew_nuts_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1037 | 20260108_towel_sweep_cashew_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1038 | 20260108_towel_sweep_cashew_nuts_from_table_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1039 | 20260108_towel_sweep_cashew_nuts_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1040 | 20260108_towel_sweep_cashew_nuts_from_table_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1041 | 20260108_towel_sweep_cashew_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1042 | 20260108_towel_sweep_cashew_nuts_from_table_34 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1043 | 20260108_towel_sweep_cashew_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1044 | 20260108_towel_sweep_cashew_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1045 | 20260108_towel_sweep_cashew_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1046 | 20260108_towel_sweep_cashew_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1047 | 20260108_towel_sweep_cashew_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1048 | 20260108_towel_sweep_cashew_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1049 | 20260108_towel_sweep_peanuts_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1050 | 20260108_towel_sweep_peanuts_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1051 | 20260108_towel_sweep_peanuts_nuts_from_table_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1052 | 20260108_towel_sweep_peanuts_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1053 | 20260108_towel_sweep_peanuts_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1054 | 20260108_towel_sweep_peanuts_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1055 | 20260108_towel_sweep_peanuts_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1056 | 20260108_towel_sweep_peanuts_nuts_from_table_16 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1057 | 20260108_towel_sweep_peanuts_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1058 | 20260108_towel_sweep_peanuts_nuts_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1059 | 20260108_towel_sweep_peanuts_nuts_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1060 | 20260108_towel_sweep_peanuts_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1061 | 20260108_towel_sweep_peanuts_nuts_from_table_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1062 | 20260108_towel_sweep_peanuts_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1063 | 20260108_towel_sweep_peanuts_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1064 | 20260108_towel_sweep_peanuts_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1065 | 20260108_towel_sweep_peanuts_nuts_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1066 | 20260108_towel_sweep_peanuts_nuts_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1067 | 20260108_towel_sweep_peanuts_nuts_from_table_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1068 | 20260108_towel_sweep_peanuts_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1087 | 20260108_towel_sweep_peanuts_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1088 | 20260108_towel_sweep_peanuts_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1089 | 20260109_scrubbrush_sweep_almond_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1092 | 20260109_scrubbrush_sweep_almond_nuts_from_table_12 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1093 | 20260109_scrubbrush_sweep_almond_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1095 | 20260109_scrubbrush_sweep_almond_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1096 | 20260109_scrubbrush_sweep_almond_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1098 | 20260109_scrubbrush_sweep_almond_nuts_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1099 | 20260109_scrubbrush_sweep_almond_nuts_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1100 | 20260109_scrubbrush_sweep_almond_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1101 | 20260109_scrubbrush_sweep_almond_nuts_from_table_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1102 | 20260109_scrubbrush_sweep_almond_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1103 | 20260109_scrubbrush_sweep_almond_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1104 | 20260109_scrubbrush_sweep_almond_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1105 | 20260109_scrubbrush_sweep_almond_nuts_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1106 | 20260109_scrubbrush_sweep_almond_nuts_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1107 | 20260109_scrubbrush_sweep_almond_nuts_from_table_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1108 | 20260109_scrubbrush_sweep_almond_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1109 | 20260109_scrubbrush_sweep_almond_nuts_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1110 | 20260109_scrubbrush_sweep_almond_nuts_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1111 | 20260109_scrubbrush_sweep_almond_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1112 | 20260109_scrubbrush_sweep_almond_nuts_from_table_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1113 | 20260109_scrubbrush_sweep_almond_nuts_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1114 | 20260109_scrubbrush_sweep_almond_nuts_from_table_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1115 | 20260109_scrubbrush_sweep_almond_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1116 | 20260109_scrubbrush_sweep_almond_nuts_from_table_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1117 | 20260109_scrubbrush_sweep_almond_nuts_from_table_35 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1119 | 20260109_scrubbrush_sweep_almond_nuts_from_table_37 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1120 | 20260109_scrubbrush_sweep_almond_nuts_from_table_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1121 | 20260109_scrubbrush_sweep_almond_nuts_from_table_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1122 | 20260109_scrubbrush_sweep_almond_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1124 | 20260109_scrubbrush_sweep_almond_nuts_from_table_41 | 操作对象无标注: cam3_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1125 | 20260109_scrubbrush_sweep_almond_nuts_from_table_42 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1127 | 20260109_scrubbrush_sweep_almond_nuts_from_table_44 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1128 | 20260109_scrubbrush_sweep_almond_nuts_from_table_45 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1129 | 20260109_scrubbrush_sweep_almond_nuts_from_table_46 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1130 | 20260109_scrubbrush_sweep_almond_nuts_from_table_47 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1131 | 20260109_scrubbrush_sweep_almond_nuts_from_table_48 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1132 | 20260109_scrubbrush_sweep_almond_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1133 | 20260109_scrubbrush_sweep_almond_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1134 | 20260109_scrubbrush_sweep_almond_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1135 | 20260109_scrubbrush_sweep_almond_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1136 | 20260109_scrubbrush_sweep_almond_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1137 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1138 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1139 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1140 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1141 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1142 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1143 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1144 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1145 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1146 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1147 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1149 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1150 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1151 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_22 | 操作对象无标注: cam6_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1152 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1153 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1154 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1155 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1156 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1157 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1158 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1159 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1160 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1161 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1162 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1163 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1164 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1165 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_35 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1166 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1167 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_37 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1168 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1169 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1170 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1171 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_40 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb |
| 1172 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_41 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1173 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1174 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1175 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1177 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1178 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1179 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1180 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1181 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1182 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1183 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1184 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1185 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1186 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1187 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1188 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1189 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1190 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1191 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1192 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1193 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1194 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1195 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1196 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1197 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1198 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1199 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1200 | 20260109_woodenbrush_sweep_peanuts_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1201 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1202 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1204 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1205 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1207 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1208 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1209 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1211 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1212 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1214 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1215 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1216 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1218 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1219 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_26 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1220 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1221 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1223 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1225 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1227 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1228 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1232 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_38 | 操作对象无标注: cam2_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1233 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1234 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1237 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_42 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1238 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_43 | 操作对象无标注: cam2_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1239 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_44 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1241 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_46 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1243 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_48 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1244 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_49 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1245 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1246 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_50 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1247 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_51 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1249 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_53 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1250 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_54 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1251 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_55 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1253 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_57 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1255 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_59 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1256 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1258 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_61 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1259 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_62 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1260 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_63 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1261 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_64 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1262 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_65 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1263 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_66 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1264 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_67 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb |
| 1265 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_68 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1266 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_69 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1267 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1268 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_70 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1269 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_71 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1270 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_72 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam4_rgb,cam5_rgb |
| 1271 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_73 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1272 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_74 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1273 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1274 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1276 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1278 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1279 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1280 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1281 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_15 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1282 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1283 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1285 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_19 | 操作对象无标注: cam6_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1286 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1288 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1289 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_22 | 操作对象无标注: cam3_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1290 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1291 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1293 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_26 | 操作对象无标注: cam6_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1294 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1295 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1296 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1298 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1300 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1301 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1303 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_35 | 操作对象无标注: cam2_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1305 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_37 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1306 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1307 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1308 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1310 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_41 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1311 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_42 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1313 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_44 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam4_rgb,cam5_rgb |
| 1314 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_45 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1315 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_46 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1316 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_47 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1318 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_49 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam3_rgb,cam4_rgb,cam5_rgb |
| 1319 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1320 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_50 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam3_rgb,cam4_rgb,cam5_rgb |
| 1322 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1323 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1324 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1326 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1327 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1329 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1330 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1331 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1333 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1334 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1335 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_18 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1336 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1339 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1340 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1342 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1343 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1345 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam7_rgb |
| 1346 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1348 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1349 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1350 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1352 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1353 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1354 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_35 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1355 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1357 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1360 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_40 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1361 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_41 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1362 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_42 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1364 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_44 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1365 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_45 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1367 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_47 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1368 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_48 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1369 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_49 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1370 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1372 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_51 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1374 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_53 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1377 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_56 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1379 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_58 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1380 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_59 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1382 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_60 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1384 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_62 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1385 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_63 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1386 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_64 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1388 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_66 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1389 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_67 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1391 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_69 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1392 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1394 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_71 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1395 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_72 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1396 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_73 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1398 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_75 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1399 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_76 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1400 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_77 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1403 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_8 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1404 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_80 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1405 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_81 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1407 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_83 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1408 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1409 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1411 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1412 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1413 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_13 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1415 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1416 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1424 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_23 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1420 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_2 | 主工具缺第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1422 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1432 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_30 | 主工具缺末帧: cam5_rgb |

</details>

---

### cc#0717
**总条数**：399 条 ｜ **完整**：57 条 ｜ **可接受(含旧UI)**：89 条

**分类统计：**

- 🔴 **全无操作对象标注**：6 条
- 🟠 **部分相机缺操作对象**：59 条
- 🟠 **操作对象缺第0帧**：35 条
- 🟠 **操作对象仅第0帧**：99 条
- 🟡 **主工具缺第0帧**：63 条
- 🟡 **主工具仅第0帧**：3 条
- 🟡 **操作对象缺100%帧**：34 条
- 🟡 **主工具缺100%帧**：11 条
- ✅ **仅缺50%帧(旧UI)**：32 条

<details>
<summary>🔴 全无操作对象标注（6 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2439 | 20260119_yellowstraw_stir_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2875 | 20260123_brush_collect_largeamount_sand_from_table_41 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2904 | 20260123_brush_collect_smallamount_sand_from_table_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2911 | 20260123_brush_collect_smallamount_sand_from_table_30 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2916 | 20260123_brush_collect_smallamount_sand_from_table_35 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam6_rgb,cam7_rgb |
| 3145 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_29 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（59 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1899 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_17 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 2385 | 20260119_greenstraw_stir_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2430 | 20260119_yellowstraw_stir_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2440 | 20260119_yellowstraw_stir_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2441 | 20260119_yellowstraw_stir_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2461 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2537 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2538 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 2539 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2540 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2542 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam2_rgb |
| 2543 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2544 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 2545 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2546 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2547 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2548 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2549 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2550 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2551 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2552 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2553 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2554 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2555 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2556 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2557 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2558 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2613 | 20260121_mallet_crush_pealed_banana_59 | 操作对象无标注: cam7_rgb |
| 2620 | 20260121_mallet_crush_pealed_banana_65 | 操作对象无标注: cam7_rgb |
| 2621 | 20260121_mallet_crush_pealed_banana_66 | 操作对象无标注: cam7_rgb; 操作对象缺末帧: cam5_rgb |
| 3330 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam2_rgb,cam6_rgb,cam7_rgb …共4项 |
| 3340 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam2_rgb |
| 3343 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3362 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3369 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3407 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3408 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_26 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3409 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3411 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3412 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam2_rgb |
| 3413 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_30 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3416 | 20260127_woodenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3424 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_50 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb |
| 3425 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_51 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam2_rgb |
| 3430 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_56 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 3434 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_60 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb |
| 3438 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_64 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 3440 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_66 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3444 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_70 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 3455 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_81 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 3458 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_84 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 3462 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_88 | 操作对象无标注: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3746 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_22 | 操作对象无标注: cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3759 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_34 | 操作对象无标注: cam2_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3761 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_36 | 操作对象无标注: cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3786 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_6 | 操作对象无标注: cam4_rgb,cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb |
| 3800 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_72 | 操作对象无标注: cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3801 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_73 | 操作对象无标注: cam4_rgb,cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam7_rgb |
| 3806 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_78 | 操作对象无标注: cam4_rgb,cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（35 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1823 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_19 | 操作对象仅第0帧: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 2208 | 20260118_pinkknife_slice_peeled_banana_56 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2324 | 20260116_mallet_crush_pealed_banana_59 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam5_rgb,cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 2329 | 20260116_mallet_crush_pealed_banana_63 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2332 | 20260116_mallet_crush_pealed_banana_66 | 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺末帧: cam7_rgb |
| 2334 | 20260116_mallet_crush_pealed_banana_68 | 操作对象缺第0帧: cam7_rgb |
| 2529 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_21 | 操作对象缺第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2612 | 20260121_mallet_crush_pealed_banana_58 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam3_rgb |
| 2615 | 20260121_mallet_crush_pealed_banana_60 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam6_rgb |
| 2617 | 20260121_mallet_crush_pealed_banana_62 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb |
| 2618 | 20260121_mallet_crush_pealed_banana_63 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam6_rgb |
| 2663 | 20260121_mallet_crush_peanuts_nuts_29 | 操作对象缺第0帧: cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2741 | 20260121_largewoodenspoon_crush_pealed_banana_15 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb |
| 2742 | 20260121_largewoodenspoon_crush_pealed_banana_16 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam5_rgb,cam6_rgb |
| 2743 | 20260121_largewoodenspoon_crush_pealed_banana_17 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam6_rgb |
| 2757 | 20260121_smallwoodenspoon_crush_pealed_banana_1 | 操作对象缺第0帧: cam7_rgb |
| 2766 | 20260121_smallwoodenspoon_crush_pealed_banana_18 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb |
| 2865 | 20260123_brush_collect_largeamount_sand_from_table_32 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2925 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_1 | 操作对象缺第0帧: cam2_rgb |
| 3005 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_43 | 操作对象缺第0帧: cam6_rgb; 主工具缺第0帧: cam6_rgb |
| 3030 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_66 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb |
| 3031 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_67 | 操作对象缺第0帧: cam7_rgb |
| 3032 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_68 | 操作对象缺第0帧: cam1_rgb,cam2_rgb |
| 3033 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_69 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam1_rgb |
| 3036 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_71 | 操作对象缺第0帧: cam2_rgb |
| 3050 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_84 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3130 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_15 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3422 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_48 | 操作对象缺第0帧: cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 3429 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_55 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb |
| 3436 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_62 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb |
| 3443 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_69 | 操作对象缺第0帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb |
| 3459 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_85 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3469 | 20260129_plasticcup_roll_dough_on_plasticcutter_15 | 操作对象缺第0帧: cam1_rgb |
| 3472 | 20260129_plasticcup_roll_dough_on_plasticcutter_18 | 操作对象缺第0帧: cam1_rgb |
| 3476 | 20260129_plasticcup_roll_dough_on_plasticcutter_21 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（99 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1557 | 20260113_book_sweep_cashew_nuts_from_table_48 | 操作对象仅第0帧: cam7_rgb |
| 1570 | 20260113_book_sweep_peanuts_nuts_from_table_16 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb |
| 1822 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_18 | 操作对象仅第0帧: cam0_rgb |
| 2876 | 20260123_brush_collect_largeamount_sand_from_table_42 | 操作对象仅第0帧: cam7_rgb |
| 2890 | 20260123_brush_collect_smallamount_sand_from_table_11 | 操作对象仅第0帧: cam7_rgb |
| 3079 | 20260126_plasticwraprod_roll_smallamount_sand_on_table_20 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3432 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_58 | 操作对象仅第0帧: cam1_rgb |
| 3724 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_10 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3725 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_100 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3726 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_101 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3727 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_102 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3728 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_103 | 操作对象仅第0帧: cam2_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3729 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_104 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3730 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_105 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3731 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_106 | 操作对象仅第0帧: cam5_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3732 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_107 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3733 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_108 | 操作对象仅第0帧: cam2_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3734 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_11 | 操作对象仅第0帧: cam7_rgb |
| 3735 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_12 | 操作对象仅第0帧: cam1_rgb,cam7_rgb |
| 3736 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_13 | 操作对象仅第0帧: cam1_rgb,cam7_rgb |
| 3739 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_16 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3740 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_17 | 操作对象仅第0帧: cam1_rgb |
| 3742 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_19 | 操作对象仅第0帧: cam5_rgb,cam7_rgb |
| 3743 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_2 | 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb |
| 3745 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_21 | 操作对象仅第0帧: cam1_rgb,cam7_rgb |
| 3747 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_23 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3748 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_24 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3749 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_25 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3750 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_26 | 操作对象仅第0帧: cam7_rgb |
| 3751 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_27 | 操作对象仅第0帧: cam7_rgb |
| 3752 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_28 | 操作对象仅第0帧: cam7_rgb |
| 3753 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_29 | 操作对象仅第0帧: cam7_rgb |
| 3755 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_30 | 操作对象仅第0帧: cam7_rgb |
| 3756 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_31 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3757 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_32 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam7_rgb |
| 3760 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_35 | 操作对象仅第0帧: cam7_rgb |
| 3762 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_37 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3763 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_38 | 操作对象仅第0帧: cam7_rgb |
| 3764 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_39 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3765 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_4 | 操作对象仅第0帧: cam0_rgb |
| 3766 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_41 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3767 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_42 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3769 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_44 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3770 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_45 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3771 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_46 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3772 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_47 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3773 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_48 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3774 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_49 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3775 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_5 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3777 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_51 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 3778 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_52 | 操作对象仅第0帧: cam7_rgb |
| 3780 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_54 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3781 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_55 | 操作对象仅第0帧: cam7_rgb |
| 3782 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_56 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3783 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_57 | 操作对象仅第0帧: cam7_rgb |
| 3784 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_58 | 操作对象仅第0帧: cam4_rgb,cam7_rgb |
| 3785 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_59 | 操作对象仅第0帧: cam5_rgb,cam7_rgb |
| 3788 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_61 | 操作对象仅第0帧: cam7_rgb |
| 3789 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_62 | 操作对象仅第0帧: cam7_rgb |
| 3791 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_64 | 操作对象仅第0帧: cam4_rgb,cam7_rgb |
| 3792 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_65 | 操作对象仅第0帧: cam0_rgb,cam4_rgb |
| 3793 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_66 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3794 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_67 | 操作对象仅第0帧: cam7_rgb |
| 3795 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_68 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3796 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_69 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3797 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_7 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3799 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_71 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3802 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_74 | 操作对象仅第0帧: cam7_rgb |
| 3803 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_75 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3804 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_76 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 3805 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_77 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3807 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_79 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3808 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_8 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3809 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_80 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3810 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_81 | 操作对象仅第0帧: cam7_rgb |
| 3811 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_82 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3829 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_99 | 操作对象仅第0帧: cam4_rgb,cam5_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3830 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_1 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3841 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_2 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3842 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_20 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3860 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_37 | 操作对象仅第0帧: cam7_rgb |
| 3861 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_38 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3929 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_35 | 操作对象仅第0帧: cam2_rgb |
| 3938 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_1 | 操作对象仅第0帧: cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3943 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_14 | 操作对象仅第0帧: cam7_rgb |
| 3946 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_17 | 操作对象仅第0帧: cam7_rgb |
| 3949 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_2 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具缺第0帧: cam0_rgb |
| 3952 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_22 | 操作对象仅第0帧: cam5_rgb; 主工具仅第0帧: cam7_rgb |
| 3956 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_26 | 操作对象仅第0帧: cam5_rgb |
| 3959 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_29 | 操作对象仅第0帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3961 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_4 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb |
| 3966 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_9 | 操作对象仅第0帧: cam3_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3968 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_10 | 操作对象仅第0帧: cam1_rgb,cam3_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 3973 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_15 | 操作对象仅第0帧: cam2_rgb,cam6_rgb |
| 3977 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_19 | 操作对象仅第0帧: cam0_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb |
| 3989 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_3 | 操作对象仅第0帧: cam1_rgb |
| 3993 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_33 | 操作对象仅第0帧: cam7_rgb |
| 3998 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_38 | 操作对象仅第0帧: cam3_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 4003 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_42 | 操作对象仅第0帧: cam6_rgb; 主工具仅第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（63 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1560 | 20260113_book_sweep_cashew_nuts_from_table_7 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam7_rgb |
| 1563 | 20260113_book_sweep_peanuts_nuts_from_table_1 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2131 | 20260118_orangeknife_slice_peeled_banana_58 | 主工具缺第0帧: cam7_rgb |
| 2139 | 20260118_orangeknife_slice_peeled_banana_65 | 主工具缺第0帧: cam7_rgb |
| 2141 | 20260118_orangeknife_slice_peeled_banana_67 | 主工具缺第0帧: cam7_rgb |
| 2142 | 20260118_orangeknife_slice_peeled_banana_68 | 主工具缺第0帧: cam7_rgb |
| 2143 | 20260118_orangeknife_slice_peeled_banana_69 | 主工具缺第0帧: cam7_rgb |
| 2145 | 20260118_orangeknife_slice_peeled_banana_70 | 主工具缺第0帧: cam0_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb |
| 2146 | 20260118_orangeknife_slice_peeled_banana_71 | 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb |
| 2149 | 20260118_orangeknife_slice_peeled_banana_74 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2290 | 20260116_mallet_crush_pealed_banana_28 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2300 | 20260116_mallet_crush_pealed_banana_37 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam3_rgb |
| 2528 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_20 | 操作对象缺末帧: cam0_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2879 | 20260123_brush_collect_largeamount_sand_from_table_45 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2880 | 20260123_brush_collect_largeamount_sand_from_table_46 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 2881 | 20260123_brush_collect_largeamount_sand_from_table_47 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 2887 | 20260123_brush_collect_sand_from_table_1 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb |
| 2888 | 20260123_brush_collect_smallamount_sand_from_table_1 | 主工具缺第0帧: cam3_rgb |
| 2894 | 20260123_brush_collect_smallamount_sand_from_table_15 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 2901 | 20260123_brush_collect_smallamount_sand_from_table_21 | 主工具缺第0帧: cam6_rgb |
| 2903 | 20260123_brush_collect_smallamount_sand_from_table_23 | 主工具缺第0帧: cam1_rgb,cam6_rgb |
| 2905 | 20260123_brush_collect_smallamount_sand_from_table_25 | 主工具缺第0帧: cam6_rgb |
| 2906 | 20260123_brush_collect_smallamount_sand_from_table_26 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 2908 | 20260123_brush_collect_smallamount_sand_from_table_28 | 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb; 主工具缺50%帧: cam6_rgb |
| 2912 | 20260123_brush_collect_smallamount_sand_from_table_31 | 操作对象缺末帧: cam1_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |
| 2927 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_100 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2928 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_101 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2929 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_102 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2930 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_103 | 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 3007 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_45 | 主工具缺第0帧: cam6_rgb |
| 3051 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_85 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3052 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_86 | 操作对象缺50%帧: cam1_rgb,cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3053 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_87 | 操作对象缺50%帧: cam1_rgb,cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3054 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_88 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb |
| 3055 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_89 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3057 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_90 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3059 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_92 | 操作对象缺50%帧: cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3060 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_93 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3061 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_94 | 操作对象缺50%帧: cam1_rgb,cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam5_rgb |
| 3062 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_95 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3063 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_96 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3064 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_97 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3065 | 20260123_plasticwraprod_roll_largeamount_sand_on_table_98 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3128 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_13 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3129 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_14 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3131 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_16 | 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam7_rgb |
| 3132 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_17 | 主工具缺第0帧: cam0_rgb |
| 3133 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_18 | 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam7_rgb |
| 3134 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_19 | 主工具缺第0帧: cam0_rgb |
| 3136 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_20 | 主工具缺第0帧: cam0_rgb |
| 3137 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_21 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 3138 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_22 | 主工具缺第0帧: cam0_rgb |
| 3139 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_23 | 主工具缺第0帧: cam0_rgb |
| 3140 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_24 | 主工具缺第0帧: cam0_rgb |
| 3423 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_49 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3492 | 20260129_plasticcup_roll_dough_on_plasticcutter_36 | 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam2_rgb |
| 3497 | 20260129_plasticcup_roll_dough_on_plasticcutter_40 | 主工具缺第0帧: cam6_rgb |
| 3498 | 20260129_plasticcup_roll_dough_on_plasticcutter_41 | 主工具缺第0帧: cam6_rgb |
| 3499 | 20260129_plasticcup_roll_dough_on_plasticcutter_42 | 主工具缺第0帧: cam6_rgb |
| 3500 | 20260129_plasticcup_roll_dough_on_plasticcutter_43 | 操作对象缺50%帧: cam6_rgb; 主工具缺第0帧: cam6_rgb |
| 3501 | 20260129_plasticcup_roll_dough_on_plasticcutter_44 | 主工具缺第0帧: cam6_rgb |
| 3503 | 20260129_plasticcup_roll_dough_on_plasticcutter_6 | 主工具缺第0帧: cam2_rgb |
| 3528 | 20260129_plasticcup_roll_largedough_on_plasticcutter_29 | 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam2_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2866 | 20260123_brush_collect_largeamount_sand_from_table_33 | 主工具仅第0帧: cam6_rgb |
| 2873 | 20260123_brush_collect_largeamount_sand_from_table_4 | 操作对象缺末帧: cam7_rgb; 主工具仅第0帧: cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 2909 | 20260123_brush_collect_smallamount_sand_from_table_29 | 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（34 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2527 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_2 | 操作对象缺末帧: cam1_rgb |
| 2531 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_4 | 操作对象缺末帧: cam7_rgb |
| 2535 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_8 | 操作对象缺末帧: cam7_rgb |
| 2738 | 20260121_largewoodenspoon_crush_pealed_banana_12 | 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2863 | 20260123_brush_collect_largeamount_sand_from_table_30 | 操作对象缺末帧: cam7_rgb |
| 2864 | 20260123_brush_collect_largeamount_sand_from_table_31 | 操作对象缺末帧: cam7_rgb |
| 2867 | 20260123_brush_collect_largeamount_sand_from_table_34 | 操作对象缺末帧: cam7_rgb |
| 2871 | 20260123_brush_collect_largeamount_sand_from_table_38 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam6_rgb,cam7_rgb |
| 2874 | 20260123_brush_collect_largeamount_sand_from_table_40 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam6_rgb,cam7_rgb |
| 2877 | 20260123_brush_collect_largeamount_sand_from_table_43 | 操作对象缺末帧: cam7_rgb |
| 2889 | 20260123_brush_collect_smallamount_sand_from_table_10 | 操作对象缺末帧: cam7_rgb |
| 2891 | 20260123_brush_collect_smallamount_sand_from_table_12 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2896 | 20260123_brush_collect_smallamount_sand_from_table_17 | 操作对象缺末帧: cam7_rgb |
| 2898 | 20260123_brush_collect_smallamount_sand_from_table_19 | 操作对象缺末帧: cam7_rgb |
| 2902 | 20260123_brush_collect_smallamount_sand_from_table_22 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam0_rgb |
| 2907 | 20260123_brush_collect_smallamount_sand_from_table_27 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb |
| 2913 | 20260123_brush_collect_smallamount_sand_from_table_32 | 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2914 | 20260123_brush_collect_smallamount_sand_from_table_33 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2915 | 20260123_brush_collect_smallamount_sand_from_table_34 | 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2917 | 20260123_brush_collect_smallamount_sand_from_table_36 | 操作对象缺末帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2918 | 20260123_brush_collect_smallamount_sand_from_table_37 | 操作对象缺末帧: cam1_rgb,cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb |
| 2919 | 20260123_brush_collect_smallamount_sand_from_table_4 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2920 | 20260123_brush_collect_smallamount_sand_from_table_5 | 操作对象缺末帧: cam7_rgb |
| 2923 | 20260123_brush_collect_smallamount_sand_from_table_8 | 操作对象缺末帧: cam7_rgb |
| 3126 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_11 | 操作对象缺末帧: cam7_rgb |
| 3127 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_12 | 操作对象缺末帧: cam7_rgb |
| 3141 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_25 | 操作对象缺末帧: cam7_rgb |
| 3142 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_26 | 操作对象缺末帧: cam7_rgb |
| 3144 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_28 | 操作对象缺末帧: cam7_rgb |
| 3146 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_3 | 操作对象缺末帧: cam7_rgb |
| 3147 | 20260123_redrubberspatula_collect_largeamount_sand_from_table_30 | 操作对象缺末帧: cam7_rgb |
| 3431 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_57 | 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb |
| 3447 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_73 | 操作对象缺末帧: cam0_rgb,cam1_rgb |
| 3452 | 20260127_ladle_stir_largeamount_coffee_in_largeshallowcontainer_78 | 操作对象缺末帧: cam1_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（11 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1821 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_17 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 2673 | 20260121_mallet_crush_peanuts_nuts_38 | 主工具缺末帧: cam7_rgb |
| 2841 | 20260123_brush_collect_largeamount_sand_from_table_10 | 操作对象缺50%帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2862 | 20260123_brush_collect_largeamount_sand_from_table_3 | 主工具缺末帧: cam7_rgb |
| 2882 | 20260123_brush_collect_largeamount_sand_from_table_5 | 主工具缺末帧: cam7_rgb |
| 2884 | 20260123_brush_collect_largeamount_sand_from_table_7 | 主工具缺末帧: cam7_rgb |
| 2886 | 20260123_brush_collect_largeamount_sand_from_table_9 | 主工具缺末帧: cam6_rgb,cam7_rgb |
| 2895 | 20260123_brush_collect_smallamount_sand_from_table_16 | 主工具缺末帧: cam7_rgb |
| 2899 | 20260123_brush_collect_smallamount_sand_from_table_2 | 主工具缺末帧: cam7_rgb |
| 2921 | 20260123_brush_collect_smallamount_sand_from_table_6 | 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb |
| 3468 | 20260129_plasticcup_roll_dough_on_plasticcutter_14 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam6_rgb |

</details>

---

### muzi#5261
**总条数**：206 条 ｜ **完整**：30 条 ｜ **可接受(含旧UI)**：38 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：93 条
- 🟠 **操作对象缺第0帧**：11 条
- 🟠 **操作对象仅第0帧**：22 条
- 🟡 **主工具缺第0帧**：12 条
- 🟡 **主工具仅第0帧**：3 条
- 🟡 **操作对象缺100%帧**：23 条
- 🟡 **主工具缺100%帧**：4 条
- ✅ **仅缺50%帧(旧UI)**：8 条

<details>
<summary>🟠 部分相机缺操作对象（93 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 413 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_20 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 419 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_26 | 操作对象无标注: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb,cam7_rgb …共4项 |
| 420 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_27 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 421 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_28 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 422 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_29 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 424 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_30 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 430 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_36 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 432 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_38 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 435 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_40 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 443 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_48 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 447 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_51 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 466 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_69 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 467 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_7 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 469 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_71 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb …共4项 |
| 470 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_72 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 471 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_73 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 478 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_8 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 479 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_80 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共6项 |
| 481 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_82 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 484 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_85 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 486 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_87 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共4项 |
| 1892 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb |
| 1893 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1894 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1979 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2409 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 2410 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2411 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2429 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2444 | 20260119_yellowstraw_stir_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2454 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2496 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2497 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2498 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2499 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2500 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2501 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2502 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2503 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam2_rgb |
| 2504 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 2505 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2506 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2507 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2508 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2509 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2510 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2511 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2512 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 2513 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2514 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2515 | 20260120_curvedwoodenspatula_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 2530 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb |
| 3328 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 4020 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4021 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4022 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4030 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam2_rgb |
| 4031 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam7_rgb |
| 4032 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam6_rgb |
| 4033 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4034 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_26 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4035 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_27 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4036 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb |
| 4038 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4039 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_30 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4040 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_31 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb |
| 4041 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_32 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam7_rgb |
| 4042 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_33 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam2_rgb |
| 4043 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_34 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4044 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_35 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4045 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_36 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4046 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_37 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4047 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_38 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4048 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_39 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4049 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4050 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_40 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb |
| 4051 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_41 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4052 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_42 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4053 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_43 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4054 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_44 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb; 主工具缺第0帧: cam2_rgb |
| 4055 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_45 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4056 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_46 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4057 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_47 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4058 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_48 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam2_rgb |
| 4059 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_49 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam2_rgb |
| 4061 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_50 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4062 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_51 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam5_rgb,cam6_rgb,cam7_rgb |
| 4063 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_52 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 4064 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_53 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |
| 4065 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4066 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4067 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4068 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（11 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 426 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_32 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 464 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_67 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 475 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_77 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1445 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_42 | 操作对象缺第0帧: cam3_rgb |
| 1813 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_1 | 操作对象缺第0帧: cam0_rgb; 主工具缺末帧: cam0_rgb; 主工具缺50%帧: cam2_rgb |
| 1853 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_1 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb |
| 2011 | 20260115_orangeknife_slice_unpealed_banana_12 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2302 | 20260116_mallet_crush_pealed_banana_39 | 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 2484 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_16 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 2490 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_4 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |
| 4028 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_20 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（22 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1454 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_50 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 3463 | 20260129_plasticcup_roll_dough_on_plasticcutter_1 | 操作对象仅第0帧: cam2_rgb,cam6_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3465 | 20260129_plasticcup_roll_dough_on_plasticcutter_11 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3520 | 20260129_plasticcup_roll_largedough_on_plasticcutter_21 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3540 | 20260129_redcup_roll_largedough_on_plasticcutter_11 | 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3549 | 20260129_redcup_roll_largedough_on_plasticcutter_2 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam1_rgb; 主工具缺末帧: cam2_rgb |
| 3550 | 20260129_redcup_roll_largedough_on_plasticcutter_20 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam1_rgb; 主工具缺末帧: cam2_rgb,cam6_rgb |
| 3554 | 20260129_redcup_roll_largedough_on_plasticcutter_24 | 操作对象仅第0帧: cam3_rgb,cam6_rgb; 主工具仅第0帧: cam6_rgb |
| 3557 | 20260129_redcup_roll_largedough_on_plasticcutter_27 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam1_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb |
| 3563 | 20260129_redcup_roll_largedough_on_plasticcutter_32 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam6_rgb |
| 3580 | 20260129_redcup_roll_largedough_on_plasticcutter_8 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具缺末帧: cam6_rgb |
| 3584 | 20260129_whitecup_roll_dough_on_plasticcutter_11 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb |
| 3587 | 20260129_whitecup_roll_dough_on_plasticcutter_14 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb |
| 3588 | 20260129_whitecup_roll_dough_on_plasticcutter_15 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb |
| 3591 | 20260129_whitecup_roll_dough_on_plasticcutter_18 | 操作对象仅第0帧: cam2_rgb,cam7_rgb; 操作对象缺末帧: cam3_rgb; 主工具缺末帧: cam6_rgb |
| 3592 | 20260129_whitecup_roll_dough_on_plasticcutter_19 | 操作对象仅第0帧: cam2_rgb; 主工具缺末帧: cam2_rgb |
| 3593 | 20260129_whitecup_roll_dough_on_plasticcutter_2 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam6_rgb |
| 3597 | 20260129_whitecup_roll_dough_on_plasticcutter_23 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam6_rgb,cam7_rgb |
| 3623 | 20260129_whitecup_roll_dough_on_plasticcutter_47 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3679 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_12 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3723 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_1 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam4_rgb |
| 3867 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_43 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam0_rgb; 主工具仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（12 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1453 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_5 | 主工具缺第0帧: cam0_rgb |
| 1506 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_16 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2270 | 20260116_mallet_crush_pealed_banana_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 2485 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_17 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2486 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_18 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2487 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_19 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2520 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_13 | 主工具缺第0帧: cam2_rgb,cam7_rgb |
| 2525 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_18 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 2526 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_19 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 3568 | 20260129_redcup_roll_largedough_on_plasticcutter_37 | 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 3572 | 20260129_redcup_roll_largedough_on_plasticcutter_40 | 操作对象缺末帧: cam2_rgb,cam3_rgb; 主工具缺第0帧: cam6_rgb |
| 3622 | 20260129_whitecup_roll_dough_on_plasticcutter_46 | 操作对象缺末帧: cam2_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam2_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1499 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_1 | 主工具仅第0帧: cam7_rgb |
| 3542 | 20260129_redcup_roll_largedough_on_plasticcutter_13 | 操作对象缺末帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3579 | 20260129_redcup_roll_largedough_on_plasticcutter_7 | 主工具仅第0帧: cam2_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（23 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 460 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_63 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 969 | 20260108_squeegee_sweep_peanuts_nuts_from_table_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1442 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_4 | 操作对象缺末帧: cam2_rgb,cam3_rgb; 操作对象缺50%帧: cam1_rgb |
| 1496 | 20260113_greenspoon_sweep_almond_nuts_from_table_7 | 操作对象缺末帧: cam7_rgb |
| 1526 | 20260113_book_sweep_cashew_nuts_from_table_2 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2008 | 20260115_orangeknife_slice_unpealed_banana_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2009 | 20260115_orangeknife_slice_unpealed_banana_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2010 | 20260115_orangeknife_slice_unpealed_banana_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2012 | 20260115_orangeknife_slice_unpealed_banana_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2013 | 20260115_orangeknife_slice_unpealed_banana_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2041 | 20260115_orangeknife_slice_unpealed_banana_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2138 | 20260118_orangeknife_slice_peeled_banana_64 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2483 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_15 | 操作对象缺末帧: cam0_rgb |
| 2493 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_7 | 操作对象缺末帧: cam7_rgb |
| 3464 | 20260129_plasticcup_roll_dough_on_plasticcutter_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3466 | 20260129_plasticcup_roll_dough_on_plasticcutter_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3527 | 20260129_plasticcup_roll_largedough_on_plasticcutter_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam3_rgb,cam4_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam5_rgb,cam7_rgb |
| 3559 | 20260129_redcup_roll_largedough_on_plasticcutter_29 | 操作对象缺末帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |
| 3561 | 20260129_redcup_roll_largedough_on_plasticcutter_30 | 操作对象缺末帧: cam2_rgb; 主工具缺末帧: cam6_rgb |
| 3585 | 20260129_whitecup_roll_dough_on_plasticcutter_12 | 操作对象缺末帧: cam2_rgb,cam3_rgb; 主工具缺末帧: cam6_rgb |
| 3586 | 20260129_whitecup_roll_dough_on_plasticcutter_13 | 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb |
| 3614 | 20260129_whitecup_roll_dough_on_plasticcutter_39 | 操作对象缺末帧: cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam2_rgb |
| 4015 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_9 | 操作对象缺末帧: cam6_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1441 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_39 | 主工具缺末帧: cam7_rgb |
| 1443 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_40 | 主工具缺末帧: cam1_rgb |
| 2522 | 20260120_redrubberspatula_stir_largeamount_coffee_shallowcontainer_15 | 主工具缺末帧: cam7_rgb |
| 3543 | 20260129_redcup_roll_largedough_on_plasticcutter_14 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam5_rgb,cam6_rgb |

</details>

---

### Qing Xinyi#19828423060
**总条数**：200 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：34 条
- 🟠 **操作对象缺第0帧**：4 条
- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **主工具缺第0帧**：28 条
- 🟡 **操作对象缺100%帧**：131 条
- 🟡 **主工具缺100%帧**：2 条

<details>
<summary>🟠 部分相机缺操作对象（34 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1922 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1932 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1935 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1943 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_30 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb …共4项 |
| 1945 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_32 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1948 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1951 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1956 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1959 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_14 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1967 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1969 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_23 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1970 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_24 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1972 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1976 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1980 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1981 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1982 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1983 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1984 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1985 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1987 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1989 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1990 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1991 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1992 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1993 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1994 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1995 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1996 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1997 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_26 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2000 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2002 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2005 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2006 | 20260114_smallwoodenspoon_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 788 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_63 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb …共4项 |
| 1235 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_40 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1917 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_7 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1974 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_5 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1965 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（28 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 862 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_61 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 863 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_62 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 864 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_63 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 866 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_65 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 867 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_66 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 868 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_67 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 870 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_69 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 875 | 20260106_squeegee_sweep_almond_nuts_from_table_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 908 | 20260106_squeegee_sweep_almond_nuts_from_table_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 919 | 20260106_squeegee_sweep_almond_nuts_from_table_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 929 | 20260108_squeegee_sweep_cashew_nuts_from_table_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 953 | 20260108_squeegee_sweep_cashew_nuts_from_table_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 954 | 20260108_squeegee_sweep_cashew_nuts_from_table_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 955 | 20260108_squeegee_sweep_cashew_nuts_from_table_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 956 | 20260108_squeegee_sweep_cashew_nuts_from_table_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 957 | 20260108_squeegee_sweep_cashew_nuts_from_table_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 958 | 20260108_squeegee_sweep_cashew_nuts_from_table_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 964 | 20260108_squeegee_sweep_peanuts_nuts_from_table_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam3_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1297 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1312 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1317 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1321 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1328 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1332 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1337 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1397 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_74 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1401 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_78 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1406 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_82 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（131 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 837 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 840 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 842 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 844 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_45 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 845 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_46 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 846 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 847 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 848 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_49 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 849 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 850 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 851 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 853 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 854 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 855 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_55 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 857 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_57 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 858 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_58 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 859 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_59 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 861 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_60 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 873 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 876 | 20260106_squeegee_sweep_almond_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 877 | 20260106_squeegee_sweep_almond_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 879 | 20260106_squeegee_sweep_almond_nuts_from_table_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 881 | 20260106_squeegee_sweep_almond_nuts_from_table_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 884 | 20260106_squeegee_sweep_almond_nuts_from_table_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 887 | 20260106_squeegee_sweep_almond_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 888 | 20260106_squeegee_sweep_almond_nuts_from_table_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 889 | 20260106_squeegee_sweep_almond_nuts_from_table_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 890 | 20260106_squeegee_sweep_almond_nuts_from_table_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 892 | 20260106_squeegee_sweep_almond_nuts_from_table_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 893 | 20260106_squeegee_sweep_almond_nuts_from_table_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 895 | 20260106_squeegee_sweep_almond_nuts_from_table_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 896 | 20260106_squeegee_sweep_almond_nuts_from_table_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 898 | 20260106_squeegee_sweep_almond_nuts_from_table_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 901 | 20260106_squeegee_sweep_almond_nuts_from_table_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 902 | 20260106_squeegee_sweep_almond_nuts_from_table_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 903 | 20260106_squeegee_sweep_almond_nuts_from_table_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 905 | 20260106_squeegee_sweep_almond_nuts_from_table_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 906 | 20260106_squeegee_sweep_almond_nuts_from_table_38 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 907 | 20260106_squeegee_sweep_almond_nuts_from_table_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 909 | 20260106_squeegee_sweep_almond_nuts_from_table_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 910 | 20260106_squeegee_sweep_almond_nuts_from_table_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 911 | 20260106_squeegee_sweep_almond_nuts_from_table_42 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 912 | 20260106_squeegee_sweep_almond_nuts_from_table_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 913 | 20260106_squeegee_sweep_almond_nuts_from_table_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 914 | 20260106_squeegee_sweep_almond_nuts_from_table_45 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 915 | 20260106_squeegee_sweep_almond_nuts_from_table_46 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 916 | 20260106_squeegee_sweep_almond_nuts_from_table_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 917 | 20260106_squeegee_sweep_almond_nuts_from_table_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 918 | 20260106_squeegee_sweep_almond_nuts_from_table_49 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 920 | 20260106_squeegee_sweep_almond_nuts_from_table_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 921 | 20260106_squeegee_sweep_almond_nuts_from_table_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 922 | 20260106_squeegee_sweep_almond_nuts_from_table_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 923 | 20260106_squeegee_sweep_almond_nuts_from_table_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 924 | 20260106_squeegee_sweep_almond_nuts_from_table_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 925 | 20260106_squeegee_sweep_almond_nuts_from_table_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 926 | 20260106_squeegee_sweep_almond_nuts_from_table_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 927 | 20260106_squeegee_sweep_almond_nuts_from_table_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 928 | 20260106_squeegee_sweep_almond_nuts_from_table_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 930 | 20260108_squeegee_sweep_cashew_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 931 | 20260108_squeegee_sweep_cashew_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 932 | 20260108_squeegee_sweep_cashew_nuts_from_table_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 933 | 20260108_squeegee_sweep_cashew_nuts_from_table_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 934 | 20260108_squeegee_sweep_cashew_nuts_from_table_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 935 | 20260108_squeegee_sweep_cashew_nuts_from_table_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 936 | 20260108_squeegee_sweep_cashew_nuts_from_table_16 | 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 937 | 20260108_squeegee_sweep_cashew_nuts_from_table_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 938 | 20260108_squeegee_sweep_cashew_nuts_from_table_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 939 | 20260108_squeegee_sweep_cashew_nuts_from_table_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 940 | 20260108_squeegee_sweep_cashew_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 941 | 20260108_squeegee_sweep_cashew_nuts_from_table_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 942 | 20260108_squeegee_sweep_cashew_nuts_from_table_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 943 | 20260108_squeegee_sweep_cashew_nuts_from_table_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 944 | 20260108_squeegee_sweep_cashew_nuts_from_table_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 945 | 20260108_squeegee_sweep_cashew_nuts_from_table_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 946 | 20260108_squeegee_sweep_cashew_nuts_from_table_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 947 | 20260108_squeegee_sweep_cashew_nuts_from_table_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 948 | 20260108_squeegee_sweep_cashew_nuts_from_table_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 949 | 20260108_squeegee_sweep_cashew_nuts_from_table_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 950 | 20260108_squeegee_sweep_cashew_nuts_from_table_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 951 | 20260108_squeegee_sweep_cashew_nuts_from_table_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 952 | 20260108_squeegee_sweep_cashew_nuts_from_table_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 959 | 20260108_squeegee_sweep_cashew_nuts_from_table_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 960 | 20260108_squeegee_sweep_cashew_nuts_from_table_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 961 | 20260108_squeegee_sweep_cashew_nuts_from_table_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 962 | 20260108_squeegee_sweep_cashew_nuts_from_table_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 963 | 20260108_squeegee_sweep_cashew_nuts_from_table_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 965 | 20260108_squeegee_sweep_peanuts_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 966 | 20260108_squeegee_sweep_peanuts_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 967 | 20260108_squeegee_sweep_peanuts_nuts_from_table_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 968 | 20260108_squeegee_sweep_peanuts_nuts_from_table_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1231 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1242 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1248 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1254 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_58 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1277 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1284 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1287 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1292 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1302 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1304 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1309 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1325 | 20260109_woodencurved_spatula_scoop_almonds_nuts_from_table_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1341 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1344 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1347 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1351 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1358 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1359 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1363 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1366 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_46 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1371 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1375 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1376 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_55 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1378 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_57 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1383 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_61 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1387 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_65 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1393 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_70 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1410 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1414 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1417 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1419 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1421 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1423 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1425 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1426 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1428 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1429 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1431 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1433 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2114 | 20260118_orangeknife_slice_peeled_banana_42 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2116 | 20260118_orangeknife_slice_peeled_banana_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 880 | 20260106_squeegee_sweep_almond_nuts_from_table_14 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 904 | 20260106_squeegee_sweep_almond_nuts_from_table_36 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |

</details>

---

### 姬子鑫#15532031882
**总条数**：187 条 ｜ **完整**：44 条 ｜ **可接受(含旧UI)**：70 条

**分类统计：**

- 🔴 **全无操作对象标注**：2 条
- 🔴 **所有标注仅第0帧**：1 条
- 🟠 **部分相机缺操作对象**：5 条
- 🟠 **操作对象缺第0帧**：30 条
- 🟠 **操作对象仅第0帧**：47 条
- 🟡 **主工具缺第0帧**：12 条
- 🟡 **主工具仅第0帧**：2 条
- 🟡 **操作对象缺100%帧**：13 条
- 🟡 **主工具缺100%帧**：5 条
- ✅ **仅缺50%帧(旧UI)**：26 条

<details>
<summary>🔴 全无操作对象标注（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2376 | 20260118_smallwoodenspoon_crush_small_peeled_banana_2 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 2664 | 20260121_mallet_crush_peanuts_nuts_3 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🔴 所有标注仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2591 | 20260121_mallet_crush_pealed_banana_39 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（5 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2340 | 20260116_mallet_crush_pealed_banana_73 | 操作对象无标注: cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam4_rgb; 操作对象缺50%帧: cam5_rgb |
| 2388 | 20260119_greenstraw_stir_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2389 | 20260119_greenstraw_stir_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2473 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2541 | 20260120_redrubberspatula_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（30 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2313 | 20260116_mallet_crush_pealed_banana_49 | 操作对象缺第0帧: cam7_rgb |
| 2322 | 20260116_mallet_crush_pealed_banana_57 | 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam7_rgb |
| 2331 | 20260116_mallet_crush_pealed_banana_65 | 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2335 | 20260116_mallet_crush_pealed_banana_69 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 2338 | 20260116_mallet_crush_pealed_banana_71 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 2339 | 20260116_mallet_crush_pealed_banana_72 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 2341 | 20260116_mallet_crush_pealed_banana_74 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2342 | 20260116_mallet_crush_pealed_banana_75 | 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 2343 | 20260116_mallet_crush_pealed_banana_76 | 操作对象缺第0帧: cam7_rgb |
| 2344 | 20260116_mallet_crush_pealed_banana_77 | 操作对象缺第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb |
| 2346 | 20260116_mallet_crush_pealed_banana_79 | 操作对象缺第0帧: cam7_rgb |
| 2349 | 20260116_mallet_crush_pealed_banana_81 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 2351 | 20260116_mallet_crush_pealed_banana_83 | 操作对象缺第0帧: cam7_rgb |
| 2356 | 20260116_mallet_crush_pealed_banana_88 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam4_rgb,cam5_rgb; 主工具缺50%帧: cam5_rgb |
| 2357 | 20260116_mallet_crush_pealed_banana_89 | 操作对象缺第0帧: cam4_rgb,cam7_rgb; 操作对象缺50%帧: cam5_rgb,cam6_rgb |
| 2359 | 20260116_mallet_crush_pealed_banana_90 | 操作对象缺第0帧: cam7_rgb |
| 2362 | 20260116_mallet_crush_pealed_banana_93 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb |
| 2363 | 20260116_mallet_crush_pealed_banana_94 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam6_rgb; 主工具缺50%帧: cam5_rgb |
| 2364 | 20260116_mallet_crush_pealed_banana_95 | 操作对象缺第0帧: cam7_rgb |
| 2386 | 20260119_greenstraw_stir_coffee_shallowcontainer_11 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2458 | 20260119_yellowstraw_stir_largeamount_coffee_shallowcontainer_13 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 2598 | 20260121_mallet_crush_pealed_banana_45 | 操作对象缺第0帧: cam0_rgb |
| 2607 | 20260121_mallet_crush_pealed_banana_53 | 操作对象缺第0帧: cam7_rgb |
| 2608 | 20260121_mallet_crush_pealed_banana_54 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam4_rgb,cam5_rgb |
| 2610 | 20260121_mallet_crush_pealed_banana_56 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb |
| 2611 | 20260121_mallet_crush_pealed_banana_57 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam1_rgb,cam4_rgb …共5项 |
| 2665 | 20260121_mallet_crush_peanuts_nuts_30 | 操作对象缺第0帧: cam4_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2675 | 20260121_mallet_crush_peanuts_nuts_4 | 操作对象缺第0帧: cam4_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3263 | 20260123_squeegee_collect_smallamount_sand_from_table_31 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3937 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_9 | 操作对象缺第0帧: cam2_rgb; 主工具缺第0帧: cam4_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（47 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2291 | 20260116_mallet_crush_pealed_banana_29 | 操作对象仅第0帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2305 | 20260116_mallet_crush_pealed_banana_41 | 操作对象仅第0帧: cam7_rgb |
| 2572 | 20260121_mallet_crush_pealed_banana_21 | 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb; 操作对象缺50%帧: cam3_rgb |
| 2579 | 20260121_mallet_crush_pealed_banana_28 | 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb; 操作对象缺50%帧: cam2_rgb,cam6_rgb |
| 2597 | 20260121_mallet_crush_pealed_banana_44 | 操作对象仅第0帧: cam0_rgb; 操作对象缺50%帧: cam1_rgb,cam5_rgb,cam6_rgb |
| 2740 | 20260121_largewoodenspoon_crush_pealed_banana_14 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam5_rgb,cam6_rgb |
| 3931 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_37 | 操作对象仅第0帧: cam2_rgb |
| 3934 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_6 | 操作对象仅第0帧: cam3_rgb,cam6_rgb |
| 3940 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3941 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 3942 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_13 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3944 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_15 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3945 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_16 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb |
| 3947 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_18 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3948 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_19 | 操作对象仅第0帧: cam2_rgb,cam4_rgb |
| 3950 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_20 | 操作对象仅第0帧: cam7_rgb |
| 3951 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3953 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_23 | 操作对象仅第0帧: cam3_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3954 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_24 | 操作对象仅第0帧: cam1_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3955 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_25 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb |
| 3957 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_27 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3958 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_28 | 操作对象仅第0帧: cam6_rgb; 主工具仅第0帧: cam0_rgb,cam4_rgb |
| 3960 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_3 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3963 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_6 | 操作对象仅第0帧: cam2_rgb,cam4_rgb,cam7_rgb |
| 3964 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_7 | 操作对象仅第0帧: cam0_rgb,cam3_rgb,cam4_rgb,cam7_rgb |
| 3965 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_8 | 操作对象仅第0帧: cam7_rgb |
| 3967 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_1 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb |
| 3969 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_11 | 操作对象仅第0帧: cam6_rgb |
| 3972 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_14 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3974 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_16 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam6_rgb |
| 3976 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_18 | 操作对象仅第0帧: cam5_rgb |
| 3981 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_22 | 操作对象仅第0帧: cam5_rgb,cam6_rgb |
| 3983 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_24 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3987 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_28 | 操作对象仅第0帧: cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3988 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3991 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_31 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam7_rgb |
| 3992 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_32 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3994 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3995 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_35 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam7_rgb |
| 3999 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_39 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 4001 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_40 | 操作对象仅第0帧: cam6_rgb,cam7_rgb |
| 4004 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_43 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 4006 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_45 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 4009 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_48 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb |
| 4010 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_49 | 操作对象仅第0帧: cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 4011 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_5 | 操作对象仅第0帧: cam2_rgb,cam3_rgb |
| 4012 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_6 | 操作对象仅第0帧: cam2_rgb,cam3_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（12 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2286 | 20260116_mallet_crush_pealed_banana_24 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2328 | 20260116_mallet_crush_pealed_banana_62 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2330 | 20260116_mallet_crush_pealed_banana_64 | 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 2347 | 20260116_mallet_crush_pealed_banana_8 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2384 | 20260119_greenstraw_stir_coffee_shallowcontainer_1 | 主工具缺第0帧: cam3_rgb |
| 2593 | 20260121_mallet_crush_pealed_banana_40 | 操作对象缺50%帧: cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2594 | 20260121_mallet_crush_pealed_banana_41 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2605 | 20260121_mallet_crush_pealed_banana_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam7_rgb |
| 3333 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_2 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 3962 | 20260128_spoonwithholes_crush_peanuts_nuts_largeplate_5 | 主工具缺第0帧: cam0_rgb |
| 3970 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_12 | 主工具缺第0帧: cam7_rgb |
| 4014 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_8 | 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2245 | 20260118_pinkknife_slice_peeled_banana_9 | 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam7_rgb |
| 3982 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_23 | 主工具仅第0帧: cam5_rgb,cam6_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（13 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2278 | 20260116_mallet_crush_pealed_banana_17 | 操作对象缺末帧: cam7_rgb |
| 2570 | 20260121_mallet_crush_pealed_banana_2 | 操作对象缺末帧: cam0_rgb,cam2_rgb; 操作对象缺50%帧: cam5_rgb,cam6_rgb,cam7_rgb |
| 2582 | 20260121_mallet_crush_pealed_banana_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2587 | 20260121_mallet_crush_pealed_banana_35 | 操作对象缺末帧: cam1_rgb |
| 2588 | 20260121_mallet_crush_pealed_banana_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb |
| 2589 | 20260121_mallet_crush_pealed_banana_37 | 操作对象缺末帧: cam0_rgb |
| 2600 | 20260121_mallet_crush_pealed_banana_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam5_rgb,cam6_rgb |
| 2601 | 20260121_mallet_crush_pealed_banana_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb |
| 2602 | 20260121_mallet_crush_pealed_banana_49 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam6_rgb; 主工具缺末帧: cam6_rgb |
| 2606 | 20260121_mallet_crush_pealed_banana_52 | 操作对象缺末帧: cam7_rgb |
| 3327 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_14 | 操作对象缺末帧: cam7_rgb |
| 3336 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_4 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 3339 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_7 | 操作对象缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（5 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2272 | 20260116_mallet_crush_pealed_banana_11 | 主工具缺末帧: cam6_rgb |
| 2336 | 20260116_mallet_crush_pealed_banana_7 | 主工具缺末帧: cam7_rgb |
| 2390 | 20260119_greenstraw_stir_coffee_shallowcontainer_15 | 主工具缺末帧: cam7_rgb |
| 2596 | 20260121_mallet_crush_pealed_banana_43 | 主工具缺末帧: cam7_rgb |
| 3323 | 20260127_greenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_10 | 主工具缺末帧: cam7_rgb |

</details>

---

### Philo#15238708815
**总条数**：164 条 ｜ **完整**：33 条 ｜ **可接受(含旧UI)**：79 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：3 条
- 🟠 **操作对象缺第0帧**：4 条
- 🟠 **操作对象仅第0帧**：15 条
- 🟡 **主工具缺第0帧**：31 条
- 🟡 **主工具仅第0帧**：2 条
- 🟡 **操作对象缺100%帧**：7 条
- 🟡 **主工具缺100%帧**：23 条
- ✅ **仅缺50%帧(旧UI)**：46 条

<details>
<summary>🟠 部分相机缺操作对象（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2316 | 20260116_mallet_crush_pealed_banana_51 | 操作对象无标注: cam7_rgb |
| 3366 | 20260127_greenchopstick_stir_smallamount_coffee_in_smallshallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb …共4项 |
| 3374 | 20260127_woodenchopstick_stir_largeamount_coffee_in_smallshallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2279 | 20260116_mallet_crush_pealed_banana_18 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam7_rgb |
| 2319 | 20260116_mallet_crush_pealed_banana_54 | 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam0_rgb,cam7_rgb |
| 2323 | 20260116_mallet_crush_pealed_banana_58 | 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam2_rgb; 操作对象缺50%帧: cam0_rgb,cam4_rgb,cam5_rgb |
| 2326 | 20260116_mallet_crush_pealed_banana_60 | 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam7_rgb …共4项 |

</details>

<details>
<summary>🟠 操作对象仅第0帧（15 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1573 | 20260113_book_sweep_peanuts_nuts_from_table_19 | 操作对象仅第0帧: cam7_rgb |
| 1579 | 20260113_book_sweep_peanuts_nuts_from_table_24 | 操作对象仅第0帧: cam2_rgb,cam6_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 2283 | 20260116_mallet_crush_pealed_banana_21 | 操作对象仅第0帧: cam1_rgb,cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 2296 | 20260116_mallet_crush_pealed_banana_33 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam3_rgb; 主工具缺末帧: cam3_rgb,cam6_rgb |
| 2312 | 20260116_mallet_crush_pealed_banana_48 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺50%帧: cam2_rgb |
| 3975 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_17 | 操作对象仅第0帧: cam3_rgb |
| 3979 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 主工具仅第0帧: cam0_rgb |
| 3984 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_25 | 操作对象仅第0帧: cam5_rgb,cam7_rgb |
| 3985 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_26 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb |
| 3990 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_30 | 操作对象仅第0帧: cam1_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |
| 3996 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam7_rgb |
| 4002 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_41 | 操作对象仅第0帧: cam4_rgb,cam5_rgb; 主工具仅第0帧: cam7_rgb |
| 4005 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_44 | 操作对象仅第0帧: cam2_rgb,cam7_rgb |
| 4008 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_47 | 操作对象仅第0帧: cam7_rgb |
| 4013 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（31 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1555 | 20260113_book_sweep_cashew_nuts_from_table_46 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam0_rgb |
| 1582 | 20260113_book_sweep_peanuts_nuts_from_table_27 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1583 | 20260113_book_sweep_peanuts_nuts_from_table_28 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 1584 | 20260113_book_sweep_peanuts_nuts_from_table_29 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1586 | 20260113_book_sweep_peanuts_nuts_from_table_30 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1587 | 20260113_book_sweep_peanuts_nuts_from_table_31 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1588 | 20260113_book_sweep_peanuts_nuts_from_table_32 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam0_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1589 | 20260113_book_sweep_peanuts_nuts_from_table_33 | 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1590 | 20260113_book_sweep_peanuts_nuts_from_table_34 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 1591 | 20260113_book_sweep_peanuts_nuts_from_table_35 | 操作对象缺50%帧: cam1_rgb,cam3_rgb,cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1596 | 20260113_book_sweep_peanuts_nuts_from_table_4 | 操作对象缺50%帧: cam5_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1785 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_24 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1786 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_25 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1787 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_26 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1788 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_27 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1789 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_28 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1790 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_29 | 主工具缺第0帧: cam3_rgb |
| 1806 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_43 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 1807 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_44 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1814 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_10 | 主工具缺第0帧: cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1816 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_12 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2150 | 20260118_orangeknife_slice_peeled_banana_75 | 主工具缺第0帧: cam7_rgb |
| 2157 | 20260118_pinkknife_slice_peeled_banana_1 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 2160 | 20260118_pinkknife_slice_peeled_banana_12 | 主工具缺第0帧: cam7_rgb |
| 2161 | 20260118_pinkknife_slice_peeled_banana_13 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2165 | 20260118_pinkknife_slice_peeled_banana_17 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam7_rgb |
| 2212 | 20260118_pinkknife_slice_peeled_banana_6 | 操作对象缺末帧: cam1_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺末帧: cam1_rgb |
| 2234 | 20260118_pinkknife_slice_peeled_banana_8 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2236 | 20260118_pinkknife_slice_peeled_banana_81 | 操作对象缺50%帧: cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam5_rgb,cam6_rgb |
| 2239 | 20260118_pinkknife_slice_peeled_banana_84 | 主工具缺第0帧: cam0_rgb |
| 2287 | 20260116_mallet_crush_pealed_banana_25 | 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1793 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_31 | 主工具仅第0帧: cam7_rgb |
| 1797 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_35 | 操作对象缺末帧: cam7_rgb; 主工具仅第0帧: cam7_rgb; 主工具缺末帧: cam6_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（7 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1561 | 20260113_book_sweep_cashew_nuts_from_table_8 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 1594 | 20260113_book_sweep_peanuts_nuts_from_table_38 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam6_rgb |
| 1796 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_34 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 2164 | 20260118_pinkknife_slice_peeled_banana_16 | 操作对象缺末帧: cam0_rgb,cam4_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb |
| 2191 | 20260118_pinkknife_slice_peeled_banana_40 | 操作对象缺末帧: cam2_rgb |
| 2273 | 20260116_mallet_crush_pealed_banana_12 | 操作对象缺末帧: cam1_rgb; 主工具缺末帧: cam1_rgb |
| 2308 | 20260116_mallet_crush_pealed_banana_44 | 操作对象缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（23 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1581 | 20260113_book_sweep_peanuts_nuts_from_table_26 | 操作对象缺50%帧: cam2_rgb,cam6_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 1776 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_16 | 主工具缺末帧: cam7_rgb |
| 1782 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_21 | 主工具缺末帧: cam7_rgb |
| 1783 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_22 | 主工具缺末帧: cam7_rgb |
| 1792 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_30 | 主工具缺末帧: cam7_rgb |
| 1794 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_32 | 主工具缺末帧: cam7_rgb |
| 1795 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_33 | 主工具缺末帧: cam7_rgb |
| 1798 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_36 | 主工具缺末帧: cam7_rgb |
| 1801 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_39 | 主工具缺末帧: cam7_rgb |
| 1803 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_40 | 主工具缺末帧: cam7_rgb |
| 1805 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_42 | 主工具缺末帧: cam7_rgb |
| 1809 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_6 | 主工具缺末帧: cam7_rgb |
| 1812 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_9 | 主工具缺末帧: cam7_rgb |
| 1820 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_16 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1824 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_2 | 主工具缺末帧: cam7_rgb |
| 1828 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_6 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb,cam6_rgb |
| 1829 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_7 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1830 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_8 | 操作对象缺50%帧: cam5_rgb; 主工具缺末帧: cam6_rgb; 主工具缺50%帧: cam5_rgb |
| 2140 | 20260118_orangeknife_slice_peeled_banana_66 | 主工具缺末帧: cam7_rgb |
| 2166 | 20260118_pinkknife_slice_peeled_banana_18 | 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 2235 | 20260118_pinkknife_slice_peeled_banana_80 | 主工具缺末帧: cam0_rgb |
| 2247 | 20260118_pinkknife_slice_peeled_banana_91 | 操作对象缺50%帧: cam5_rgb; 主工具缺末帧: cam0_rgb; 主工具缺50%帧: cam5_rgb |
| 2250 | 20260118_pinkknife_slice_peeled_banana_94 | 主工具缺末帧: cam0_rgb |

</details>

---

### CJH#13166938010
**总条数**：152 条 ｜ **完整**：2 条 ｜ **可接受(含旧UI)**：11 条

**分类统计：**

- 🟠 **操作对象缺第0帧**：2 条
- 🟠 **操作对象仅第0帧**：40 条
- 🟡 **主工具缺第0帧**：31 条
- 🟡 **操作对象缺100%帧**：68 条
- ✅ **仅缺50%帧(旧UI)**：9 条

<details>
<summary>🟠 操作对象缺第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 20 | 20260104_largeplate_mallet_flatten_crush_largedough_27 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 43 | 20260104_smallplate_mallet_flatten_crush_smalldough_17 | 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺末帧: cam0_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（40 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 36 | 20260104_smallplate_mallet_flatten_crush_smalldough_10 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 39 | 20260104_smallplate_mallet_flatten_crush_smalldough_13 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 66 | 20260104_smallplate_mallet_flatten_crush_smalldough_4 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb …共4项 |
| 67 | 20260104_smallplate_mallet_flatten_crush_smalldough_5 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb …共4项 |
| 3546 | 20260129_redcup_roll_largedough_on_plasticcutter_17 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3553 | 20260129_redcup_roll_largedough_on_plasticcutter_23 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3555 | 20260129_redcup_roll_largedough_on_plasticcutter_25 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam6_rgb |
| 3562 | 20260129_redcup_roll_largedough_on_plasticcutter_31 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3564 | 20260129_redcup_roll_largedough_on_plasticcutter_33 | 操作对象仅第0帧: cam2_rgb |
| 3565 | 20260129_redcup_roll_largedough_on_plasticcutter_34 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam6_rgb |
| 3566 | 20260129_redcup_roll_largedough_on_plasticcutter_35 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam6_rgb |
| 3575 | 20260129_redcup_roll_largedough_on_plasticcutter_43 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb |
| 3578 | 20260129_redcup_roll_largedough_on_plasticcutter_6 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam6_rgb |
| 3589 | 20260129_whitecup_roll_dough_on_plasticcutter_16 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3590 | 20260129_whitecup_roll_dough_on_plasticcutter_17 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3594 | 20260129_whitecup_roll_dough_on_plasticcutter_20 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具仅第0帧: cam6_rgb |
| 3595 | 20260129_whitecup_roll_dough_on_plasticcutter_21 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb |
| 3596 | 20260129_whitecup_roll_dough_on_plasticcutter_22 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3598 | 20260129_whitecup_roll_dough_on_plasticcutter_24 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb |
| 3599 | 20260129_whitecup_roll_dough_on_plasticcutter_25 | 操作对象仅第0帧: cam2_rgb,cam6_rgb,cam7_rgb |
| 3600 | 20260129_whitecup_roll_dough_on_plasticcutter_26 | 操作对象仅第0帧: cam2_rgb,cam3_rgb |
| 3601 | 20260129_whitecup_roll_dough_on_plasticcutter_27 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3602 | 20260129_whitecup_roll_dough_on_plasticcutter_28 | 操作对象仅第0帧: cam2_rgb,cam3_rgb |
| 3603 | 20260129_whitecup_roll_dough_on_plasticcutter_29 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb |
| 3604 | 20260129_whitecup_roll_dough_on_plasticcutter_3 | 操作对象仅第0帧: cam2_rgb,cam3_rgb; 主工具仅第0帧: cam6_rgb |
| 3605 | 20260129_whitecup_roll_dough_on_plasticcutter_30 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb |
| 3606 | 20260129_whitecup_roll_dough_on_plasticcutter_31 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam2_rgb |
| 3607 | 20260129_whitecup_roll_dough_on_plasticcutter_32 | 操作对象仅第0帧: cam2_rgb |
| 3609 | 20260129_whitecup_roll_dough_on_plasticcutter_34 | 操作对象仅第0帧: cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam6_rgb |
| 3611 | 20260129_whitecup_roll_dough_on_plasticcutter_36 | 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam7_rgb |
| 3612 | 20260129_whitecup_roll_dough_on_plasticcutter_37 | 操作对象仅第0帧: cam2_rgb,cam6_rgb |
| 3831 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_10 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3833 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_12 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3834 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_13 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3835 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_14 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3836 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_15 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3837 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_16 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3838 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_17 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3839 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_18 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3900 | 20260128_bigwoodenspoon_crush_almond_nuts_largeplate_7 | 操作对象仅第0帧: cam2_rgb; 主工具仅第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（31 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 15 | 20260104_largeplate_mallet_flatten_crush_largedough_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 16 | 20260104_largeplate_mallet_flatten_crush_largedough_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 17 | 20260104_largeplate_mallet_flatten_crush_largedough_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 18 | 20260104_largeplate_mallet_flatten_crush_largedough_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 19 | 20260104_largeplate_mallet_flatten_crush_largedough_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 53 | 20260104_smallplate_mallet_flatten_crush_smalldough_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 54 | 20260104_smallplate_mallet_flatten_crush_smalldough_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 55 | 20260104_smallplate_mallet_flatten_crush_smalldough_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 56 | 20260104_smallplate_mallet_flatten_crush_smalldough_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 57 | 20260104_smallplate_mallet_flatten_crush_smalldough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 58 | 20260104_smallplate_mallet_flatten_crush_smalldough_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 62 | 20260104_smallplate_mallet_flatten_crush_smalldough_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 63 | 20260104_smallplate_mallet_flatten_crush_smalldough_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 72 | 20260105_largeplate_greenchopstick_slice_largedough_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 73 | 20260105_largeplate_greenchopstick_slice_largedough_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 82 | 20260105_largeplate_greenchopstick_slice_largedough_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 83 | 20260105_largeplate_greenchopstick_slice_largedough_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 92 | 20260105_largeplate_greenchopstick_slice_largedough_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 93 | 20260105_largeplate_greenchopstick_slice_largedough_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 95 | 20260105_largeplate_greenchopstick_slice_largedough_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 98 | 20260105_largeplate_greenchopstick_slice_largedough_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 100 | 20260105_largeplate_greenchopstick_slice_largedough_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 102 | 20260105_largeplate_greenchopstick_slice_largedough_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb …共4项 |
| 103 | 20260105_largeplate_woodenchopstick_slice_largedough_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 114 | 20260105_largeplate_woodenchopstick_slice_largedough_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 122 | 20260105_largeplate_woodenchopstick_slice_largedough_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 123 | 20260105_largeplate_woodenchopstick_slice_largedough_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 124 | 20260105_largeplate_woodenchopstick_slice_largedough_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 125 | 20260105_largeplate_woodenchopstick_slice_largedough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam4_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 126 | 20260105_largeplate_woodenchopstick_slice_largedough_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 127 | 20260105_largeplate_woodenchopstick_slice_largedough_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam5_rgb …共4项 |

</details>

<details>
<summary>🟡 操作对象缺100%帧（68 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 7 | 20260104_largeplate_mallet_flatten_crush_largedough_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 10 | 20260104_largeplate_mallet_flatten_crush_largedough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 11 | 20260104_largeplate_mallet_flatten_crush_largedough_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 12 | 20260104_largeplate_mallet_flatten_crush_largedough_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 13 | 20260104_largeplate_mallet_flatten_crush_largedough_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 14 | 20260104_largeplate_mallet_flatten_crush_largedough_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 21 | 20260104_largeplate_mallet_flatten_crush_largedough_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 22 | 20260104_largeplate_mallet_flatten_crush_largedough_29 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 23 | 20260104_largeplate_mallet_flatten_crush_largedough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 24 | 20260104_largeplate_mallet_flatten_crush_largedough_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 25 | 20260104_largeplate_mallet_flatten_crush_largedough_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 26 | 20260104_largeplate_mallet_flatten_crush_largedough_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 27 | 20260104_largeplate_mallet_flatten_crush_largedough_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 28 | 20260104_largeplate_mallet_flatten_crush_largedough_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 29 | 20260104_largeplate_mallet_flatten_crush_largedough_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 30 | 20260104_largeplate_mallet_flatten_crush_largedough_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 31 | 20260104_largeplate_mallet_flatten_crush_largedough_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 33 | 20260104_largeplate_mallet_flatten_crush_largedough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 34 | 20260104_largeplate_mallet_flatten_crush_largedough_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 35 | 20260104_smallplate_mallet_flatten_crush_smalldough_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 37 | 20260104_smallplate_mallet_flatten_crush_smalldough_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 38 | 20260104_smallplate_mallet_flatten_crush_smalldough_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 40 | 20260104_smallplate_mallet_flatten_crush_smalldough_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 41 | 20260104_smallplate_mallet_flatten_crush_smalldough_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 42 | 20260104_smallplate_mallet_flatten_crush_smalldough_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 44 | 20260104_smallplate_mallet_flatten_crush_smalldough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 45 | 20260104_smallplate_mallet_flatten_crush_smalldough_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 46 | 20260104_smallplate_mallet_flatten_crush_smalldough_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 47 | 20260104_smallplate_mallet_flatten_crush_smalldough_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 48 | 20260104_smallplate_mallet_flatten_crush_smalldough_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 49 | 20260104_smallplate_mallet_flatten_crush_smalldough_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 50 | 20260104_smallplate_mallet_flatten_crush_smalldough_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 51 | 20260104_smallplate_mallet_flatten_crush_smalldough_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 52 | 20260104_smallplate_mallet_flatten_crush_smalldough_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 60 | 20260104_smallplate_mallet_flatten_crush_smalldough_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 61 | 20260104_smallplate_mallet_flatten_crush_smalldough_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 64 | 20260104_smallplate_mallet_flatten_crush_smalldough_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 65 | 20260104_smallplate_mallet_flatten_crush_smalldough_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 68 | 20260104_smallplate_mallet_flatten_crush_smalldough_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 69 | 20260104_smallplate_mallet_flatten_crush_smalldough_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 70 | 20260104_smallplate_mallet_flatten_crush_smalldough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 71 | 20260104_smallplate_mallet_flatten_crush_smalldough_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 75 | 20260105_largeplate_greenchopstick_slice_largedough_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 76 | 20260105_largeplate_greenchopstick_slice_largedough_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 85 | 20260105_largeplate_greenchopstick_slice_largedough_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 86 | 20260105_largeplate_greenchopstick_slice_largedough_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 87 | 20260105_largeplate_greenchopstick_slice_largedough_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 89 | 20260105_largeplate_greenchopstick_slice_largedough_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 90 | 20260105_largeplate_greenchopstick_slice_largedough_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 96 | 20260105_largeplate_greenchopstick_slice_largedough_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 104 | 20260105_largeplate_woodenchopstick_slice_largedough_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 105 | 20260105_largeplate_woodenchopstick_slice_largedough_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 106 | 20260105_largeplate_woodenchopstick_slice_largedough_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 107 | 20260105_largeplate_woodenchopstick_slice_largedough_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 108 | 20260105_largeplate_woodenchopstick_slice_largedough_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 110 | 20260105_largeplate_woodenchopstick_slice_largedough_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 111 | 20260105_largeplate_woodenchopstick_slice_largedough_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 113 | 20260105_largeplate_woodenchopstick_slice_largedough_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 116 | 20260105_largeplate_woodenchopstick_slice_largedough_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 117 | 20260105_largeplate_woodenchopstick_slice_largedough_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 118 | 20260105_largeplate_woodenchopstick_slice_largedough_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 119 | 20260105_largeplate_woodenchopstick_slice_largedough_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 120 | 20260105_largeplate_woodenchopstick_slice_largedough_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 121 | 20260105_largeplate_woodenchopstick_slice_largedough_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 128 | 20260105_largeplate_woodenchopstick_slice_largedough_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 129 | 20260105_largeplate_woodenchopstick_slice_largedough_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 130 | 20260105_largeplate_woodenchopstick_slice_largedough_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 235 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_22 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### 闫珈旭#19500360304
**总条数**：131 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🔴 **所有标注仅第0帧**：126 条
- 🟠 **部分相机缺操作对象**：2 条
- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **操作对象缺100%帧**：2 条

<details>
<summary>🔴 所有标注仅第0帧（126 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 380 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_79 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 381 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_8 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 382 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_80 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 391 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_89 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 393 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_90 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 395 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_92 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 396 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_93 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 397 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_94 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 398 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_95 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 400 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_97 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 402 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 403 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 404 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 406 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_14 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 407 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 408 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 409 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 410 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_18 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 411 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 412 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_2 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 414 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_21 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 415 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 416 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 417 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 418 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_25 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 423 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 425 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_31 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 433 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_39 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 434 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_4 | 操作对象无标注: cam0_rgb,cam7_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 436 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_41 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 437 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_42 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 438 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_43 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 439 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_44 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 440 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_45 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 441 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_46 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 442 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_47 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 444 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_49 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 445 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 446 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_50 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 448 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_52 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 449 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_53 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 450 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_54 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 452 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_56 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象仅第0帧: cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 453 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_57 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 454 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_58 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 455 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_59 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 456 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_6 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 457 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_60 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 458 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_61 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 459 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_62 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 499 | 20260104_smallplate_woodenchopstick_slice_smalldough_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 504 | 20260104_smallplate_woodenchopstick_slice_smalldough_24 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 506 | 20260104_smallplate_woodenchopstick_slice_smalldough_26 | 操作对象无标注: cam7_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 507 | 20260104_smallplate_woodenchopstick_slice_smalldough_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 508 | 20260104_smallplate_woodenchopstick_slice_smalldough_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 509 | 20260104_smallplate_woodenchopstick_slice_smalldough_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 511 | 20260104_smallplate_woodenchopstick_slice_smalldough_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 513 | 20260104_smallplate_woodenchopstick_slice_smalldough_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 514 | 20260104_smallplate_woodenchopstick_slice_smalldough_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 515 | 20260104_smallplate_woodenchopstick_slice_smalldough_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 517 | 20260104_smallplate_woodenchopstick_slice_smalldough_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 518 | 20260104_smallplate_woodenchopstick_slice_smalldough_37 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 519 | 20260104_smallplate_woodenchopstick_slice_smalldough_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 520 | 20260104_smallplate_woodenchopstick_slice_smalldough_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 521 | 20260104_smallplate_woodenchopstick_slice_smalldough_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 522 | 20260104_smallplate_woodenchopstick_slice_smalldough_40 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 523 | 20260104_smallplate_woodenchopstick_slice_smalldough_41 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 524 | 20260104_smallplate_woodenchopstick_slice_smalldough_42 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 525 | 20260104_smallplate_woodenchopstick_slice_smalldough_43 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 526 | 20260104_smallplate_woodenchopstick_slice_smalldough_44 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 528 | 20260104_smallplate_woodenchopstick_slice_smalldough_46 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 529 | 20260104_smallplate_woodenchopstick_slice_smalldough_47 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 531 | 20260104_smallplate_woodenchopstick_slice_smalldough_49 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 532 | 20260104_smallplate_woodenchopstick_slice_smalldough_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb |
| 534 | 20260104_smallplate_woodenchopstick_slice_smalldough_51 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 536 | 20260104_smallplate_woodenchopstick_slice_smalldough_53 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 537 | 20260104_smallplate_woodenchopstick_slice_smalldough_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 539 | 20260104_smallplate_woodenchopstick_slice_smalldough_9 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 540 | 20260105_smallplate_greenchopstick_slice_smalldough_1 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 542 | 20260105_smallplate_greenchopstick_slice_smalldough_11 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 543 | 20260105_smallplate_greenchopstick_slice_smalldough_12 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 545 | 20260105_smallplate_greenchopstick_slice_smalldough_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 546 | 20260105_smallplate_greenchopstick_slice_smalldough_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 547 | 20260105_smallplate_greenchopstick_slice_smalldough_16 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 550 | 20260105_smallplate_greenchopstick_slice_smalldough_19 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 551 | 20260105_smallplate_greenchopstick_slice_smalldough_2 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 552 | 20260105_smallplate_greenchopstick_slice_smalldough_20 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 553 | 20260105_smallplate_greenchopstick_slice_smalldough_21 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 554 | 20260105_smallplate_greenchopstick_slice_smalldough_22 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 555 | 20260105_smallplate_greenchopstick_slice_smalldough_23 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 557 | 20260105_smallplate_greenchopstick_slice_smalldough_25 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 559 | 20260105_smallplate_greenchopstick_slice_smalldough_27 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 560 | 20260105_smallplate_greenchopstick_slice_smalldough_28 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 561 | 20260105_smallplate_greenchopstick_slice_smalldough_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam5_rgb,cam7_rgb |
| 564 | 20260105_smallplate_greenchopstick_slice_smalldough_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 565 | 20260105_smallplate_greenchopstick_slice_smalldough_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam5_rgb,cam7_rgb |
| 566 | 20260105_smallplate_greenchopstick_slice_smalldough_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 567 | 20260105_smallplate_greenchopstick_slice_smalldough_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 569 | 20260105_smallplate_greenchopstick_slice_smalldough_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam5_rgb,cam6_rgb |
| 1069 | 20260108_towel_sweep_peanuts_nuts_from_table_28 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1070 | 20260108_towel_sweep_peanuts_nuts_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1071 | 20260108_towel_sweep_peanuts_nuts_from_table_3 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1073 | 20260108_towel_sweep_peanuts_nuts_from_table_31 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1074 | 20260108_towel_sweep_peanuts_nuts_from_table_32 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1075 | 20260108_towel_sweep_peanuts_nuts_from_table_33 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1076 | 20260108_towel_sweep_peanuts_nuts_from_table_34 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1077 | 20260108_towel_sweep_peanuts_nuts_from_table_35 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1078 | 20260108_towel_sweep_peanuts_nuts_from_table_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1079 | 20260108_towel_sweep_peanuts_nuts_from_table_37 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1080 | 20260108_towel_sweep_peanuts_nuts_from_table_38 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1081 | 20260108_towel_sweep_peanuts_nuts_from_table_39 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1083 | 20260108_towel_sweep_peanuts_nuts_from_table_40 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1084 | 20260108_towel_sweep_peanuts_nuts_from_table_5 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1085 | 20260108_towel_sweep_peanuts_nuts_from_table_6 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1090 | 20260109_scrubbrush_sweep_almond_nuts_from_table_10 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1094 | 20260109_scrubbrush_sweep_almond_nuts_from_table_14 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1097 | 20260109_scrubbrush_sweep_almond_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1126 | 20260109_scrubbrush_sweep_almond_nuts_from_table_43 | 操作对象无标注: cam6_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1148 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_2 | 操作对象无标注: cam6_rgb; 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1523 | 20260113_book_sweep_cashew_nuts_from_table_17 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1536 | 20260113_book_sweep_cashew_nuts_from_table_29 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1544 | 20260113_book_sweep_cashew_nuts_from_table_36 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1548 | 20260113_book_sweep_cashew_nuts_from_table_4 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 1549 | 20260113_book_sweep_cashew_nuts_from_table_40 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1556 | 20260113_book_sweep_cashew_nuts_from_table_47 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 2252 | 20260118_pinkknife_slice_unpealed_banana_30 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 394 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_91 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 操作对象缺第0帧: cam7_rgb …共4项 |
| 1551 | 20260113_book_sweep_cashew_nuts_from_table_42 | 操作对象无标注: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 3699 | 20260130_rollingpin_roll_small_dough_on_cuttingboard_30 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam5_rgb,cam7_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb; 主工具仅第0帧: cam0_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1543 | 20260113_book_sweep_cashew_nuts_from_table_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1559 | 20260113_book_sweep_cashew_nuts_from_table_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### Aria#18600244282
**总条数**：113 条 ｜ **完整**：21 条 ｜ **可接受(含旧UI)**：35 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：19 条
- 🟠 **操作对象缺第0帧**：19 条
- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **主工具缺第0帧**：17 条
- 🟡 **操作对象缺100%帧**：22 条
- ✅ **仅缺50%帧(旧UI)**：14 条

<details>
<summary>🟠 部分相机缺操作对象（19 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 583 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 585 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 594 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 596 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 599 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_26 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 603 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_3 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 607 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_33 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 609 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_35 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 612 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_38 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 613 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_39 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 618 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_43 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 621 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_46 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 624 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_49 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 626 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_50 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 629 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_53 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 638 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_61 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 645 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_68 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb …共4项 |
| 654 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_76 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 790 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_65 | 操作对象无标注: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb …共4项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（19 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 587 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_15 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 593 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_20 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 729 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_1 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb …共5项 |
| 730 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_10 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 747 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_26 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 754 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_32 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 755 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_33 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 758 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_36 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 765 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_42 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 772 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_49 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 800 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_74 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 804 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_9 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 1891 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_1 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb |
| 2014 | 20260115_orangeknife_slice_unpealed_banana_15 | 操作对象缺第0帧: cam7_rgb |
| 2345 | 20260116_mallet_crush_pealed_banana_78 | 操作对象缺第0帧: cam7_rgb |
| 2348 | 20260116_mallet_crush_pealed_banana_80 | 操作对象缺第0帧: cam7_rgb |
| 2352 | 20260116_mallet_crush_pealed_banana_84 | 操作对象缺第0帧: cam7_rgb |
| 2353 | 20260116_mallet_crush_pealed_banana_85 | 操作对象缺第0帧: cam0_rgb,cam7_rgb |
| 2355 | 20260116_mallet_crush_pealed_banana_87 | 操作对象缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2027 | 20260115_orangeknife_slice_unpealed_banana_27 | 操作对象仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（17 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 491 | 20260104_smallplate_woodenchopstick_slice_smalldough_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 495 | 20260104_smallplate_woodenchopstick_slice_smalldough_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 538 | 20260104_smallplate_woodenchopstick_slice_smalldough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 544 | 20260105_smallplate_greenchopstick_slice_smalldough_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 549 | 20260105_smallplate_greenchopstick_slice_smalldough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 558 | 20260105_smallplate_greenchopstick_slice_smalldough_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 570 | 20260105_smallplate_greenchopstick_slice_smalldough_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 671 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 869 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_68 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 872 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_70 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1203 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_11 | 主工具缺第0帧: cam0_rgb |
| 1402 | 20260113_redrubberspatula_scoop_peanuts_nuts_from_table_79 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2135 | 20260118_orangeknife_slice_peeled_banana_61 | 主工具缺第0帧: cam7_rgb |
| 2147 | 20260118_orangeknife_slice_peeled_banana_72 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 2159 | 20260118_pinkknife_slice_peeled_banana_11 | 主工具缺第0帧: cam7_rgb |
| 2162 | 20260118_pinkknife_slice_peeled_banana_14 | 主工具缺第0帧: cam7_rgb |
| 2190 | 20260118_pinkknife_slice_peeled_banana_4 | 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（22 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 497 | 20260104_smallplate_woodenchopstick_slice_smalldough_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 516 | 20260104_smallplate_woodenchopstick_slice_smalldough_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 530 | 20260104_smallplate_woodenchopstick_slice_smalldough_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 535 | 20260104_smallplate_woodenchopstick_slice_smalldough_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 562 | 20260105_smallplate_greenchopstick_slice_smalldough_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 576 | 20260105_smallplate_greenchopstick_slice_smalldough_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 580 | 20260105_smallplate_greenchopstick_slice_smalldough_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 736 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 738 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 759 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 777 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 778 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 799 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_73 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 814 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 815 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 829 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 831 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 841 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_42 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 843 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 871 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 897 | 20260106_squeegee_sweep_almond_nuts_from_table_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1252 | 20260109_redrubber_spatula_scoop_cashew_nuts_from_table_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### kk#15294568167
**总条数**：106 条 ｜ **完整**：13 条 ｜ **可接受(含旧UI)**：20 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：71 条
- 🟠 **操作对象缺第0帧**：3 条
- 🟡 **主工具缺第0帧**：5 条
- 🟡 **操作对象缺100%帧**：6 条
- 🟡 **主工具缺100%帧**：1 条
- ✅ **仅缺50%帧(旧UI)**：7 条

<details>
<summary>🟠 部分相机缺操作对象（71 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 134 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_1 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 135 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 136 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_11 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 137 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb …共6项 |
| 138 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 139 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_14 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 140 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_15 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺末帧: cam5_rgb,cam6_rgb …共5项 |
| 141 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 142 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_17 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 143 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_18 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺末帧: cam5_rgb,cam6_rgb …共5项 |
| 144 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 145 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_2 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 146 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb …共6项 |
| 147 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_21 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 148 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 149 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_23 | 操作对象无标注: cam0_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 150 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 151 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_25 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 152 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_26 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb,cam6_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb …共5项 |
| 153 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_27 | 操作对象无标注: cam0_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb …共6项 |
| 154 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_28 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam3_rgb,cam7_rgb …共6项 |
| 155 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb …共6项 |
| 156 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb …共5项 |
| 157 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_30 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb …共6项 |
| 158 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 159 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_32 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 160 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_33 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 161 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_34 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 162 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_35 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 163 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_36 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 164 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_37 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 165 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 166 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_39 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb,cam7_rgb …共6项 |
| 167 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 168 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_40 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 169 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_41 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 170 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_42 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象仅第0帧: cam3_rgb,cam4_rgb; 操作对象缺第0帧: cam7_rgb …共5项 |
| 171 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_43 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb,cam7_rgb …共6项 |
| 172 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_44 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 173 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_45 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 174 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_46 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 175 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_47 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb …共6项 |
| 176 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_48 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 177 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_49 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 178 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_5 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 179 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_50 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 180 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_51 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam3_rgb …共6项 |
| 1839 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1851 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 1874 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb |
| 1875 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam0_rgb |
| 1878 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 1880 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1886 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1888 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_7 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1889 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1890 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb |
| 1895 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1900 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb |
| 1901 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1903 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1906 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_23 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1930 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb |
| 1937 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 1941 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 1947 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_34 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb; 操作对象缺50%帧: cam7_rgb |
| 1952 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1960 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1971 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_25 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1973 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1977 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_8 | 操作对象无标注: cam0_rgb,cam1_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1975 | 20260114_smallwoodenspoon_stir_largeamount_coffee_shallowcontainer_6 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |
| 2163 | 20260118_pinkknife_slice_peeled_banana_15 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam3_rgb |
| 2735 | 20260121_largewoodenspoon_crush_pealed_banana_1 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（5 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2151 | 20260118_orangeknife_slice_peeled_banana_76 | 主工具缺第0帧: cam7_rgb; 主工具缺末帧: cam0_rgb |
| 2152 | 20260118_orangeknife_slice_peeled_banana_77 | 主工具缺第0帧: cam0_rgb |
| 2158 | 20260118_pinkknife_slice_peeled_banana_10 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam7_rgb |
| 2197 | 20260118_pinkknife_slice_peeled_banana_46 | 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2201 | 20260118_pinkknife_slice_peeled_banana_5 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（6 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 131 | 20260105_largeplate_woodenchopstick_slice_largedough_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 132 | 20260105_largeplate_woodenchopstick_slice_largedough_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 133 | 20260105_largeplate_woodenchopstick_slice_largedough_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1656 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1657 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2180 | 20260118_pinkknife_slice_peeled_banana_30 | 操作对象缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2288 | 20260116_mallet_crush_pealed_banana_26 | 操作对象缺50%帧: cam7_rgb; 主工具缺末帧: cam7_rgb |

</details>

---

### 张含羽#19030879790
**总条数**：99 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🔴 **全无操作对象标注**：1 条
- 🟠 **部分相机缺操作对象**：39 条
- 🟠 **操作对象缺第0帧**：23 条
- 🟠 **操作对象仅第0帧**：15 条
- 🟡 **主工具缺第0帧**：6 条
- 🟡 **操作对象缺100%帧**：15 条

<details>
<summary>🔴 全无操作对象标注（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 886 | 20260106_squeegee_sweep_almond_nuts_from_table_2 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（39 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 312 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_17 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 313 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_18 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 319 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_23 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 320 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_24 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 322 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_26 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 323 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_27 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 324 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_28 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 326 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_3 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 328 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_31 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 330 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_33 | 操作对象无标注: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 331 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_34 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 332 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_35 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 333 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_36 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 334 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_37 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 335 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_38 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 337 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_4 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 341 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_43 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 364 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_64 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 378 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_77 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共4项 |
| 401 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 451 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_55 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 461 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_64 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 462 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_65 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 465 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_68 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 468 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_70 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 472 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_74 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 476 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_78 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 480 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_81 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb …共4项 |
| 482 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_83 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 483 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_84 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb …共4项 |
| 581 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 660 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_1 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共6项 |
| 1864 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 2315 | 20260116_mallet_crush_pealed_banana_50 | 操作对象无标注: cam7_rgb; 操作对象缺第0帧: cam0_rgb,cam4_rgb,cam5_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb …共4项 |
| 2350 | 20260116_mallet_crush_pealed_banana_82 | 操作对象无标注: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2477 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_1 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 4023 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 4037 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_29 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam7_rgb |
| 4060 | 20260130_bigwoodenspoon_press_sponge_in_largeshallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam5_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（23 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 309 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_14 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 311 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_16 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 315 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_2 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 316 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_20 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 317 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_21 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 318 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_22 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 321 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_25 | 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 325 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_29 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 327 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_30 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 336 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_39 | 操作对象仅第0帧: cam1_rgb; 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 339 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_41 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 343 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_45 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 345 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_47 | 操作对象缺第0帧: cam1_rgb,cam2_rgb; 操作对象缺末帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam5_rgb,cam7_rgb …共4项 |
| 363 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_63 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 366 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_66 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 368 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_68 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 372 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_71 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共6项 |
| 463 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_66 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 489 | 20260104_smallplate_woodenchopstick_slice_smalldough_10 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共5项 |
| 2479 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_11 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3477 | 20260129_plasticcup_roll_dough_on_plasticcutter_22 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3535 | 20260129_plasticcup_roll_largedough_on_plasticcutter_7 | 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 3821 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_91 | 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam5_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（15 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 310 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_15 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 329 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_32 | 操作对象仅第0帧: cam0_rgb,cam2_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 342 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_44 | 操作对象仅第0帧: cam0_rgb,cam2_rgb; 操作对象缺末帧: cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2055 | 20260115_pinkknife_slice_unpealed_banana_15 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2318 | 20260116_mallet_crush_pealed_banana_53 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3538 | 20260129_redcup_roll_largedough_on_plasticcutter_1 | 操作对象仅第0帧: cam1_rgb,cam2_rgb,cam3_rgb; 主工具仅第0帧: cam2_rgb |
| 3560 | 20260129_redcup_roll_largedough_on_plasticcutter_3 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb; 主工具仅第0帧: cam2_rgb …共4项 |
| 3574 | 20260129_redcup_roll_largedough_on_plasticcutter_42 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam3_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共4项 |
| 3621 | 20260129_whitecup_roll_dough_on_plasticcutter_45 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb; 主工具缺末帧: cam2_rgb |
| 3636 | 20260130_rollingpin_roll_large_dough_on_cuttingboard_12 | 操作对象仅第0帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb |
| 3768 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_43 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb …共4项 |
| 3776 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_50 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3832 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_11 | 操作对象仅第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam0_rgb; 主工具仅第0帧: cam7_rgb |
| 3855 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_32 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam4_rgb; 主工具缺末帧: cam4_rgb,cam7_rgb |
| 3882 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_57 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（6 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 488 | 20260104_smallplate_woodenchopstick_slice_smalldough_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 490 | 20260104_smallplate_woodenchopstick_slice_smalldough_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 494 | 20260104_smallplate_woodenchopstick_slice_smalldough_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 496 | 20260104_smallplate_woodenchopstick_slice_smalldough_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 3567 | 20260129_redcup_roll_largedough_on_plasticcutter_36 | 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb |
| 3787 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_60 | 操作对象缺末帧: cam2_rgb,cam3_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（15 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 308 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 314 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 512 | 20260104_smallplate_woodenchopstick_slice_smalldough_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 852 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 883 | 20260106_squeegee_sweep_almond_nuts_from_table_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 885 | 20260106_squeegee_sweep_almond_nuts_from_table_19 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 972 | 20260108_squeegee_sweep_peanuts_nuts_from_table_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1648 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2102 | 20260118_orangeknife_slice_peeled_banana_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2294 | 20260116_mallet_crush_pealed_banana_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2478 | 20260120_curvedwoodenspatula_stir_largeamount_coffee_shallowcontainer_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3779 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_53 | 操作对象缺末帧: cam4_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3790 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_63 | 操作对象缺末帧: cam2_rgb,cam3_rgb |
| 3815 | 20260129_plasticwraprollrod_roll_small_dough_on_cuttingboard_86 | 操作对象缺末帧: cam2_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 3870 | 20260130_plasticwraprollrod_roll_large_dough_on_cuttingboard_46 | 操作对象缺末帧: cam0_rgb,cam4_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |

</details>

---

### RQ#1111
**总条数**：78 条 ｜ **完整**：8 条 ｜ **可接受(含旧UI)**：27 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：23 条
- 🟠 **操作对象缺第0帧**：4 条
- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **主工具缺第0帧**：5 条
- 🟡 **操作对象缺100%帧**：4 条
- 🟡 **主工具缺100%帧**：14 条
- ✅ **仅缺50%帧(旧UI)**：19 条

<details>
<summary>🟠 部分相机缺操作对象（23 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1869 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1871 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1872 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 1873 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1876 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb |
| 1877 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb |
| 1881 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb |
| 1882 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1883 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1884 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1885 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1887 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1897 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1898 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_16 | 操作对象无标注: cam1_rgb; 操作对象缺第0帧: cam0_rgb |
| 1902 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1905 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1907 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_24 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1911 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_28 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb; 主工具缺第0帧: cam0_rgb |
| 1920 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1927 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_16 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1934 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_22 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1938 | 20260114_largewoodenspoon_stir_smallamount_coffee_shallowcontainer_26 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam7_rgb |
| 2408 | 20260119_greenstraw_stir_largeamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1850 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_7 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1896 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_14 | 操作对象缺第0帧: cam1_rgb; 操作对象缺50%帧: cam0_rgb |
| 1909 | 20260114_largewoodenspoon_stir_largeamount_coffee_shallowcontainer_26 | 操作对象缺第0帧: cam0_rgb,cam1_rgb |
| 2354 | 20260116_mallet_crush_pealed_banana_86 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam1_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1646 | 20260113_book_sweep_peanuts_nuts_from_table_9 | 操作对象仅第0帧: cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam6_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（5 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 59 | 20260104_smallplate_mallet_flatten_crush_smalldough_31 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1619 | 20260113_book_sweep_peanuts_nuts_from_table_60 | 主工具缺第0帧: cam6_rgb |
| 1781 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_20 | 主工具缺第0帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2136 | 20260118_orangeknife_slice_peeled_banana_62 | 主工具缺第0帧: cam7_rgb |
| 2137 | 20260118_orangeknife_slice_peeled_banana_63 | 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1606 | 20260113_book_sweep_peanuts_nuts_from_table_49 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam6_rgb |
| 1638 | 20260113_book_sweep_peanuts_nuts_from_table_78 | 操作对象缺末帧: cam1_rgb; 操作对象缺50%帧: cam5_rgb,cam7_rgb |
| 1860 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_16 | 操作对象缺末帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 1861 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_2 | 操作对象缺末帧: cam0_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（14 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1 | 20260104_largeplate_mallet_flatten_crush_largedough_1 | 操作对象缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb |
| 1176 | 20260109_woodenbrush_sweep_peanuts_cashew_from_table_8 | 主工具缺末帧: cam7_rgb |
| 1672 | 20260114_largewoodenspoon_largeplate_cut_kineticsand_32 | 主工具缺末帧: cam7_rgb |
| 1771 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_11 | 主工具缺末帧: cam7_rgb |
| 1772 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_12 | 主工具缺末帧: cam7_rgb |
| 1773 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_13 | 主工具缺末帧: cam6_rgb,cam7_rgb |
| 1779 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_19 | 主工具缺末帧: cam7_rgb |
| 1780 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_2 | 主工具缺末帧: cam7_rgb |
| 1784 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_23 | 主工具缺末帧: cam7_rgb |
| 1856 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_12 | 主工具缺末帧: cam7_rgb |
| 1858 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_14 | 主工具缺末帧: cam7_rgb |
| 1863 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_4 | 主工具缺末帧: cam7_rgb |
| 1866 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_7 | 主工具缺末帧: cam7_rgb |
| 1867 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_8 | 操作对象缺50%帧: cam0_rgb; 主工具缺末帧: cam7_rgb |

</details>

---

### Sijia Li#13377653788
**总条数**：60 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：9 条

**分类统计：**

- 🔴 **全无操作对象标注**：36 条
- 🟠 **操作对象仅第0帧**：2 条
- 🟡 **主工具缺第0帧**：7 条
- 🟡 **操作对象缺100%帧**：2 条
- 🟡 **主工具缺100%帧**：4 条
- ✅ **仅缺50%帧(旧UI)**：9 条

<details>
<summary>🔴 全无操作对象标注（36 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1007 | 20260108_towel_sweep_almond_nuts_from_table_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺末帧: cam1_rgb …共4项 |
| 1519 | 20260113_book_sweep_cashew_nuts_from_table_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1520 | 20260113_book_sweep_cashew_nuts_from_table_14 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1521 | 20260113_book_sweep_cashew_nuts_from_table_15 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1522 | 20260113_book_sweep_cashew_nuts_from_table_16 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1524 | 20260113_book_sweep_cashew_nuts_from_table_18 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1525 | 20260113_book_sweep_cashew_nuts_from_table_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1527 | 20260113_book_sweep_cashew_nuts_from_table_20 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1528 | 20260113_book_sweep_cashew_nuts_from_table_21 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1529 | 20260113_book_sweep_cashew_nuts_from_table_22 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1530 | 20260113_book_sweep_cashew_nuts_from_table_23 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1531 | 20260113_book_sweep_cashew_nuts_from_table_24 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1532 | 20260113_book_sweep_cashew_nuts_from_table_25 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1533 | 20260113_book_sweep_cashew_nuts_from_table_26 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1534 | 20260113_book_sweep_cashew_nuts_from_table_27 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb |
| 1535 | 20260113_book_sweep_cashew_nuts_from_table_28 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1538 | 20260113_book_sweep_cashew_nuts_from_table_30 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 1539 | 20260113_book_sweep_cashew_nuts_from_table_31 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1540 | 20260113_book_sweep_cashew_nuts_from_table_32 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1541 | 20260113_book_sweep_cashew_nuts_from_table_33 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1542 | 20260113_book_sweep_cashew_nuts_from_table_34 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 1545 | 20260113_book_sweep_cashew_nuts_from_table_37 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1546 | 20260113_book_sweep_cashew_nuts_from_table_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1547 | 20260113_book_sweep_cashew_nuts_from_table_39 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam6_rgb |
| 1804 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_41 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1810 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_7 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 1811 | 20260113_woodenspoon_sweep_cashew_nuts_from_table_8 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 1815 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1817 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1818 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1819 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1854 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb |
| 1879 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 2652 | 20260121_mallet_crush_peanuts_nuts_19 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2654 | 20260121_mallet_crush_peanuts_nuts_20 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2668 | 20260121_mallet_crush_peanuts_nuts_33 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 3909 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_17 | 操作对象仅第0帧: cam2_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 3986 | 20260130_smallwoodenspoon_crush_peanuts_nuts_largeplate_27 | 操作对象仅第0帧: cam1_rgb,cam3_rgb; 主工具仅第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（7 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 976 | 20260108_squeegee_sweep_peanuts_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 979 | 20260108_squeegee_sweep_peanuts_nuts_from_table_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 983 | 20260108_squeegee_sweep_peanuts_nuts_from_table_27 | 主工具缺第0帧: cam3_rgb |
| 985 | 20260108_squeegee_sweep_peanuts_nuts_from_table_4 | 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 987 | 20260108_squeegee_sweep_peanuts_nuts_from_table_6 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1005 | 20260108_towel_sweep_almond_nuts_from_table_22 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 2670 | 20260121_mallet_crush_peanuts_nuts_35 | 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 995 | 20260108_towel_sweep_almond_nuts_from_table_13 | 操作对象缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1518 | 20260113_book_sweep_cashew_nuts_from_table_12 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 977 | 20260108_squeegee_sweep_peanuts_nuts_from_table_21 | 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 988 | 20260108_squeegee_sweep_peanuts_nuts_from_table_7 | 主工具缺末帧: cam3_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 989 | 20260108_squeegee_sweep_peanuts_nuts_from_table_8 | 操作对象缺50%帧: cam2_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1008 | 20260108_towel_sweep_almond_nuts_from_table_3 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |

</details>

---

### Zhang Jiali#18611666863
**总条数**：60 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🔴 **全无操作对象标注**：1 条
- 🟠 **部分相机缺操作对象**：3 条
- 🟠 **操作对象缺第0帧**：4 条
- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **主工具缺第0帧**：7 条
- 🟡 **主工具仅第0帧**：1 条
- 🟡 **操作对象缺100%帧**：43 条

<details>
<summary>🔴 全无操作对象标注（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 264 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_49 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 部分相机缺操作对象（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 305 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_10 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb; 操作对象缺第0帧: cam7_rgb …共7项 |
| 306 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_11 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 307 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_12 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam1_rgb …共5项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 276 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_6 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 277 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_60 | 操作对象缺第0帧: cam0_rgb,cam7_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 278 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_61 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 304 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_1 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |

</details>

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 287 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_7 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（7 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 258 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_43 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 297 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_79 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 299 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_80 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 300 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_81 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 301 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_82 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb …共4项 |
| 302 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_83 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 303 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 262 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_47 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（43 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 241 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_28 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 243 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_3 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 245 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 246 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 248 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_34 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 249 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_35 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 250 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_36 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 251 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_37 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 253 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_39 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 255 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 259 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_44 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 260 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_45 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 261 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_46 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 263 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_48 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 265 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_5 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 266 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_50 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam4_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 267 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_51 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 268 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_52 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 269 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_53 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 270 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_54 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 271 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_55 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 272 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 273 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_57 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 274 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_58 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 275 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_59 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 279 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_62 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 280 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_63 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 281 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_64 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 282 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_65 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 283 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_66 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 284 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_67 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 285 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_68 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 286 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_69 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 288 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_70 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 289 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_71 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 290 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_72 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 291 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_73 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 292 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_74 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 293 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_75 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 294 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_76 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 295 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_77 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 296 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_78 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 298 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_8 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### Shirley Klein#17797254369
**总条数**：50 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：37 条
- 🟠 **操作对象缺第0帧**：1 条
- 🟡 **主工具缺第0帧**：1 条
- 🟡 **操作对象缺100%帧**：11 条

<details>
<summary>🟠 部分相机缺操作对象（37 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 183 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_54 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 184 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_55 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象仅第0帧: cam2_rgb,cam7_rgb; 操作对象缺末帧: cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |
| 186 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_57 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 187 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_58 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 188 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_59 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 189 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_6 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb …共4项 |
| 190 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_60 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 191 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_61 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 192 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_62 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb …共6项 |
| 193 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_63 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 194 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_64 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 195 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_65 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 196 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_66 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb …共6项 |
| 197 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_67 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 198 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_68 | 操作对象无标注: cam0_rgb,cam4_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 199 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_69 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb …共5项 |
| 200 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_7 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam3_rgb …共7项 |
| 201 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_70 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 202 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_71 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 203 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_72 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam7_rgb; 操作对象仅第0帧: cam4_rgb; 操作对象缺第0帧: cam3_rgb …共6项 |
| 204 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_73 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb,cam4_rgb; 操作对象缺末帧: cam3_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 205 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_74 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 206 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_75 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 207 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_76 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb …共4项 |
| 208 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_77 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb …共6项 |
| 209 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_78 | 操作对象无标注: cam0_rgb,cam1_rgb,cam4_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共5项 |
| 210 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_79 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam1_rgb,cam4_rgb …共6项 |
| 211 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam4_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb …共6项 |
| 212 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_80 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb; 操作对象缺第0帧: cam3_rgb,cam7_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 213 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_81 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb …共5项 |
| 214 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_82 | 操作对象无标注: cam0_rgb; 操作对象仅第0帧: cam2_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb …共6项 |
| 215 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_83 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 216 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_84 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 217 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_85 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象仅第0帧: cam2_rgb; 操作对象缺第0帧: cam4_rgb …共6项 |
| 218 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_86 | 操作对象无标注: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam6_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb …共5项 |
| 219 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_87 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam2_rgb,cam3_rgb,cam7_rgb; 操作对象缺末帧: cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 220 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_9 | 操作对象无标注: cam0_rgb,cam2_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 242 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_29 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 221 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb,cam4_rgb; 主工具缺末帧: cam7_rgb …共4项 |

</details>

<details>
<summary>🟡 操作对象缺100%帧（11 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 185 | 20260105_icecreamscooper_scoop_cashew_nuts_from_deepbowl_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 222 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 223 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 224 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 226 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_14 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 230 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 236 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 238 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_25 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam4_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam5_rgb,cam6_rgb |
| 239 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_26 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 254 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_4 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 257 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_42 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |

</details>

---

### tys#18537168165
**总条数**：50 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：12 条
- 🟠 **操作对象缺第0帧**：11 条
- 🟠 **操作对象仅第0帧**：9 条
- 🟡 **主工具缺第0帧**：3 条
- 🟡 **操作对象缺100%帧**：15 条

<details>
<summary>🟠 部分相机缺操作对象（12 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 252 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_38 | 操作对象无标注: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb; 操作对象缺第0帧: cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 340 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_42 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 344 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_46 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |
| 347 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_49 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 348 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_5 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb …共4项 |
| 349 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_50 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 351 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_52 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 352 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_53 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 355 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_56 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 361 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_61 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 362 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_62 | 操作对象无标注: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 377 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_76 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam3_rgb,cam7_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（11 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 228 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_16 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 231 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_19 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 346 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_48 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 350 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_51 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 353 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_54 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 354 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_55 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 356 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_57 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 357 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_58 | 操作对象仅第0帧: cam7_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 374 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_73 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 375 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_74 | 操作对象仅第0帧: cam0_rgb,cam7_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |
| 390 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_88 | 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam1_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共5项 |

</details>

<details>
<summary>🟠 操作对象仅第0帧（9 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 359 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_6 | 操作对象仅第0帧: cam2_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 360 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_60 | 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 365 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_65 | 操作对象仅第0帧: cam0_rgb,cam1_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 367 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_67 | 操作对象仅第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 369 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_69 | 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 373 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_72 | 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 376 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_75 | 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 379 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_78 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 387 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_85 | 操作对象仅第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb …共4项 |

</details>

<details>
<summary>🟡 主工具缺第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 227 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_15 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 370 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam3_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 385 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_83 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（15 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 225 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 229 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_17 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 232 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 233 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 234 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_21 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 237 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 240 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 244 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_30 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 247 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 256 | 20260105_icescoop_scoop_almond_nuts_from_blackpan_41 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 338 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_40 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 358 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_59 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 371 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_70 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 383 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_81 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 392 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_9 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### C hen yu#15852997385
**总条数**：49 条 ｜ **完整**：12 条 ｜ **可接受(含旧UI)**：13 条

**分类统计：**

- 🟠 **操作对象仅第0帧**：1 条
- 🟡 **主工具缺第0帧**：28 条
- 🟡 **主工具仅第0帧**：1 条
- 🟡 **操作对象缺100%帧**：2 条
- 🟡 **主工具缺100%帧**：4 条
- ✅ **仅缺50%帧(旧UI)**：1 条

<details>
<summary>🟠 操作对象仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1465 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_60 | 操作对象仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（28 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1460 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_56 | 操作对象缺50%帧: cam2_rgb; 主工具缺第0帧: cam3_rgb |
| 1461 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_57 | 主工具缺第0帧: cam3_rgb |
| 1462 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_58 | 主工具缺第0帧: cam3_rgb |
| 1464 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_6 | 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 1469 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_64 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1470 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_65 | 主工具缺第0帧: cam1_rgb,cam2_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1471 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_66 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1472 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_67 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1473 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_68 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1474 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_69 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1475 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_7 | 主工具缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 1476 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_70 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam2_rgb,cam6_rgb,cam7_rgb |
| 1477 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_71 | 操作对象缺50%帧: cam0_rgb,cam4_rgb; 主工具缺第0帧: cam1_rgb,cam2_rgb,cam6_rgb,cam7_rgb |
| 1478 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_72 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam1_rgb,cam2_rgb,cam6_rgb,cam7_rgb |
| 1479 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_73 | 主工具缺第0帧: cam1_rgb,cam6_rgb,cam7_rgb |
| 1480 | 20260113_woodenspatula_scoop_cashew_nuts_from_table_8 | 主工具缺第0帧: cam0_rgb |
| 1482 | 20260113_greenspoon_sweep_almond_nuts_from_table_1 | 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1489 | 20260113_greenspoon_sweep_almond_nuts_from_table_16 | 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb |
| 1490 | 20260113_greenspoon_sweep_almond_nuts_from_table_17 | 主工具缺第0帧: cam0_rgb,cam4_rgb,cam6_rgb,cam7_rgb |
| 1491 | 20260113_greenspoon_sweep_almond_nuts_from_table_2 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam3_rgb; 主工具缺末帧: cam7_rgb |
| 1492 | 20260113_greenspoon_sweep_almond_nuts_from_table_3 | 主工具缺第0帧: cam0_rgb,cam3_rgb,cam4_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 1493 | 20260113_greenspoon_sweep_almond_nuts_from_table_4 | 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam3_rgb; 主工具缺末帧: cam6_rgb,cam7_rgb |
| 1494 | 20260113_greenspoon_sweep_almond_nuts_from_table_5 | 主工具缺第0帧: cam0_rgb,cam4_rgb |
| 1495 | 20260113_greenspoon_sweep_almond_nuts_from_table_6 | 主工具缺第0帧: cam1_rgb |
| 1500 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_10 | 主工具缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam7_rgb |
| 1505 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_15 | 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1513 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_8 | 主工具缺第0帧: cam5_rgb |
| 1515 | 20260113_book_sweep_cashew_nuts_from_table_1 | 操作对象缺50%帧: cam2_rgb,cam6_rgb; 主工具缺第0帧: cam0_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1511 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_6 | 主工具仅第0帧: cam7_rgb; 主工具缺50%帧: cam6_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1512 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_7 | 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1516 | 20260113_book_sweep_cashew_nuts_from_table_10 | 操作对象缺末帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb,cam6_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1486 | 20260113_greenspoon_sweep_almond_nuts_from_table_13 | 主工具缺末帧: cam7_rgb |
| 1504 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_14 | 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam3_rgb |
| 1508 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_3 | 主工具缺末帧: cam7_rgb |
| 1509 | 20260113_smallwoodenspoon_sweep_almond_nuts_from_table_4 | 主工具缺末帧: cam7_rgb |

</details>

---

### bear#19866267196
**总条数**：49 条 ｜ **完整**：12 条 ｜ **可接受(含旧UI)**：12 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：24 条
- 🟠 **操作对象缺第0帧**：8 条
- 🟡 **主工具缺第0帧**：1 条
- 🟡 **操作对象缺100%帧**：1 条
- 🟡 **主工具缺100%帧**：3 条

<details>
<summary>🟠 部分相机缺操作对象（24 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1831 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1832 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_1 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1833 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1835 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_12 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1836 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_13 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 1837 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_14 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam1_rgb |
| 1838 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_15 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1840 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_17 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam2_rgb |
| 1841 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_18 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1842 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_19 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb,cam3_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 1843 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_2 | 操作对象无标注: cam0_rgb,cam1_rgb |
| 1844 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_20 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam7_rgb …共5项 |
| 1845 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_21 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb,cam7_rgb |
| 1846 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺末帧: cam7_rgb |
| 1847 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_4 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺50%帧: cam2_rgb |
| 1848 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_5 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺第0帧: cam0_rgb |
| 1849 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_6 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1852 | 20260115_greenfork_stir_smallamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1855 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_11 | 操作对象无标注: cam0_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1862 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_3 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 1868 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_9 | 操作对象无标注: cam0_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺50%帧: cam1_rgb |
| 1870 | 20260115_woodenfork_stir_smallamount_coffee_shallowcontainer_10 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺50%帧: cam2_rgb |
| 2292 | 20260116_mallet_crush_pealed_banana_3 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam3_rgb,cam4_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam7_rgb |
| 2311 | 20260116_mallet_crush_pealed_banana_47 | 操作对象无标注: cam2_rgb,cam6_rgb |

</details>

<details>
<summary>🟠 操作对象缺第0帧（8 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1827 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_5 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb |
| 1857 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_13 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |
| 1865 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_6 | 操作对象缺第0帧: cam0_rgb; 主工具缺末帧: cam7_rgb |
| 2230 | 20260118_pinkknife_slice_peeled_banana_76 | 操作对象缺第0帧: cam7_rgb; 主工具缺50%帧: cam1_rgb |
| 2317 | 20260116_mallet_crush_pealed_banana_52 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam4_rgb,cam5_rgb,cam7_rgb |
| 2320 | 20260116_mallet_crush_pealed_banana_55 | 操作对象缺第0帧: cam2_rgb |
| 2333 | 20260116_mallet_crush_pealed_banana_67 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb,cam1_rgb,cam4_rgb,cam5_rgb |
| 2337 | 20260116_mallet_crush_pealed_banana_70 | 操作对象缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2327 | 20260116_mallet_crush_pealed_banana_61 | 操作对象缺50%帧: cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1825 | 20260115_greenfork_stir_largeamount_coffee_shallowcontainer_3 | 操作对象缺末帧: cam0_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam2_rgb,cam6_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1859 | 20260115_woodenfork_stir_largeamount_coffee_shallowcontainer_15 | 操作对象缺50%帧: cam1_rgb; 主工具缺末帧: cam7_rgb |
| 2227 | 20260118_pinkknife_slice_peeled_banana_73 | 主工具缺末帧: cam0_rgb |
| 2241 | 20260118_pinkknife_slice_peeled_banana_86 | 主工具缺末帧: cam0_rgb |

</details>

---

### Xu#0989
**总条数**：33 条 ｜ **完整**：3 条 ｜ **可接受(含旧UI)**：10 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：2 条
- 🟠 **操作对象缺第0帧**：3 条
- 🟡 **主工具缺第0帧**：2 条
- 🟡 **主工具仅第0帧**：1 条
- 🟡 **操作对象缺100%帧**：15 条
- ✅ **仅缺50%帧(旧UI)**：7 条

<details>
<summary>🟠 部分相机缺操作对象（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 651 | 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_73 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam6_rgb,cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb …共4项 |
| 727 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_8 | 操作对象无标注: cam0_rgb,cam1_rgb; 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb …共4项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 672 | 20260106_metalspoon_scoop_cashew_nuts_from_transparent_bowl_20 | 操作对象缺第0帧: cam0_rgb,cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 746 | 20260106_plasticspoon_scoop_almond_nuts_from_whitepan_25 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb …共4项 |
| 874 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_9 | 操作对象缺第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 865 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_64 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 1003 | 20260108_towel_sweep_almond_nuts_from_table_20 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |

</details>

<details>
<summary>🟡 主工具仅第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 1000 | 20260108_towel_sweep_almond_nuts_from_table_18 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具仅第0帧: cam7_rgb; 主工具缺末帧: cam6_rgb …共4项 |

</details>

<details>
<summary>🟡 操作对象缺100%帧（15 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 856 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_56 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 860 | 20260106_woodenspoon_scoop_peanut_nuts_from_blackpan_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 878 | 20260106_squeegee_sweep_almond_nuts_from_table_12 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 882 | 20260106_squeegee_sweep_almond_nuts_from_table_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 891 | 20260106_squeegee_sweep_almond_nuts_from_table_24 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 894 | 20260106_squeegee_sweep_almond_nuts_from_table_27 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 899 | 20260106_squeegee_sweep_almond_nuts_from_table_31 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 900 | 20260106_squeegee_sweep_almond_nuts_from_table_32 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1002 | 20260108_towel_sweep_almond_nuts_from_table_2 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1006 | 20260108_towel_sweep_almond_nuts_from_table_23 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1011 | 20260108_towel_sweep_almond_nuts_from_table_6 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1012 | 20260108_towel_sweep_almond_nuts_from_table_7 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1015 | 20260108_towel_sweep_cashew_nuts_from_table_1 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 1016 | 20260108_towel_sweep_cashew_nuts_from_table_10 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |
| 1017 | 20260108_towel_sweep_cashew_nuts_from_table_11 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺末帧: cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb |

</details>

---

### jia#18323005301
**总条数**：20 条 ｜ **完整**：7 条 ｜ **可接受(含旧UI)**：11 条

**分类统计：**

- 🟠 **操作对象缺第0帧**：3 条
- 🟠 **操作对象仅第0帧**：2 条
- 🟡 **主工具缺第0帧**：2 条
- 🟡 **操作对象缺100%帧**：1 条
- 🟡 **主工具缺100%帧**：1 条
- ✅ **仅缺50%帧(旧UI)**：4 条

<details>
<summary>🟠 操作对象缺第0帧（3 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2274 | 20260116_mallet_crush_pealed_banana_13 | 操作对象缺第0帧: cam2_rgb; 操作对象缺末帧: cam7_rgb |
| 2298 | 20260116_mallet_crush_pealed_banana_35 | 操作对象缺第0帧: cam2_rgb,cam6_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 2304 | 20260116_mallet_crush_pealed_banana_40 | 操作对象缺第0帧: cam7_rgb; 操作对象缺50%帧: cam0_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2285 | 20260116_mallet_crush_pealed_banana_23 | 操作对象仅第0帧: cam7_rgb; 操作对象缺50%帧: cam2_rgb |
| 2293 | 20260116_mallet_crush_pealed_banana_30 | 操作对象仅第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2289 | 20260116_mallet_crush_pealed_banana_27 | 操作对象缺50%帧: cam2_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam7_rgb; 主工具缺末帧: cam6_rgb |
| 2314 | 20260116_mallet_crush_pealed_banana_5 | 主工具缺第0帧: cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2280 | 20260116_mallet_crush_pealed_banana_19 | 操作对象缺末帧: cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2297 | 20260116_mallet_crush_pealed_banana_34 | 操作对象缺50%帧: cam3_rgb,cam7_rgb; 主工具缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### HAHA#18651017066
**总条数**：10 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：5 条
- 🟠 **操作对象缺第0帧**：2 条
- 🟡 **主工具缺第0帧**：2 条
- 🟡 **操作对象缺100%帧**：1 条

<details>
<summary>🟠 部分相机缺操作对象（5 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 388 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_86 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺第0帧: cam7_rgb …共6项 |
| 428 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_34 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 429 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_35 | 操作对象无标注: cam0_rgb; 操作对象缺50%帧: cam1_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 431 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_37 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共4项 |
| 477 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_79 | 操作对象无标注: cam0_rgb; 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb …共5项 |

</details>

<details>
<summary>🟠 操作对象缺第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 389 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_87 | 操作对象缺第0帧: cam0_rgb; 操作对象缺末帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb,cam7_rgb …共4项 |
| 405 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_13 | 操作对象缺第0帧: cam1_rgb; 操作对象缺末帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 主工具缺第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 384 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_82 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam6_rgb; 主工具缺50%帧: cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |
| 386 | 20260105_measuringscoop_scoop_almond_nuts_from_transparentbowl_84 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺第0帧: cam0_rgb,cam1_rgb,cam6_rgb; 主工具缺50%帧: cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 427 | 20260105_openscoop_scoop_peanuts_nuts_from_whitepan_33 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### Tan Mengdie#15173533596
**总条数**：10 条 ｜ **完整**：5 条 ｜ **可接受(含旧UI)**：5 条

**分类统计：**

- 🟠 **部分相机缺操作对象**：1 条
- 🟠 **操作对象仅第0帧**：4 条

<details>
<summary>🟠 部分相机缺操作对象（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 3932 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_4 | 操作对象无标注: cam1_rgb; 操作对象仅第0帧: cam0_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟠 操作对象仅第0帧（4 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 3922 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_29 | 操作对象仅第0帧: cam2_rgb |
| 3925 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_31 | 操作对象仅第0帧: cam6_rgb |
| 3928 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_34 | 操作对象仅第0帧: cam3_rgb |
| 3930 | 20260128_bigwoodenspoon_crush_peanuts_nuts_largeplate_36 | 操作对象仅第0帧: cam1_rgb,cam2_rgb |

</details>

---

### Liu Rundong#17318648128
**总条数**：3 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟠 **操作对象缺第0帧**：1 条
- 🟡 **操作对象缺100%帧**：2 条

<details>
<summary>🟠 操作对象缺第0帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 6 | 20260104_largeplate_mallet_flatten_crush_largedough_14 | 操作对象缺第0帧: cam7_rgb; 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

<details>
<summary>🟡 操作对象缺100%帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 5 | 20260104_largeplate_mallet_flatten_crush_largedough_13 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 8 | 20260104_largeplate_mallet_flatten_crush_largedough_16 | 操作对象缺末帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb; 主工具缺50%帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### rq
**总条数**：2 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟡 **主工具仅第0帧**：2 条

<details>
<summary>🟡 主工具仅第0帧（2 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2 | 20260104_largeplate_mallet_flatten_crush_largedough_10 | 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |
| 3 | 20260104_largeplate_mallet_flatten_crush_largedough_11 | 主工具仅第0帧: cam0_rgb,cam1_rgb,cam2_rgb,cam3_rgb,cam4_rgb,cam5_rgb,cam6_rgb,cam7_rgb |

</details>

---

### Kehan Li#0000
**总条数**：1 条 ｜ **完整**：0 条 ｜ **可接受(含旧UI)**：0 条

**分类统计：**

- 🟡 **主工具缺100%帧**：1 条

<details>
<summary>🟡 主工具缺100%帧（1 条）</summary>

| ID | 实验名 | 缺漏详情 |
|-----|--------|---------|
| 2387 | 20260119_greenstraw_stir_coffee_shallowcontainer_12 | 主工具缺末帧: cam3_rgb,cam7_rgb |

</details>

---
