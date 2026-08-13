# Step 4 — factorial ablation of the noncommittal flag

All five arms scored in a single run, 3 passes each, arm order randomised within each row. Every arm is compared against the **control arm's rerun**, paired per row — never against the stored baseline score, because the judge is stochastic at temperature 0.3.

- rows: **56**
- scored row-arm-passes: **657**
- judge calls: **1971**

## Control arm — does the baseline zero reproduce?

All 56 rows are baseline zeros. Control-arm P(zero) over 3 passes:

| P(zero) | rows |
|---|---|
| 1.00 | 49 |
| 0.67 | 4 |
| 0.33 | 1 |
| 0.00 | 2 |

Mean control P(zero) = **0.929**. This is the reference every arm below is measured against.

## Arm comparisons vs control

| arm | role | n paired | mean P(zero) | control P(zero) | delta | Wilcoxon p | mean cosine | control cosine |
|---|---|---|---|---|---|---|---|---|
| `-H` | PRIMARY | 38 | 0.851 | 0.956 | -0.105 | 0.04156 | 0.8919 | 0.9106 |
| `-Q` | secondary | 47 | 0.496 | 0.915 | -0.418 | 1.22e-05 | 0.9215 | 0.9176 |
| `-D` | negative control | 25 | 0.800 | 0.973 | -0.173 | 0.03564 | 0.9226 | 0.9224 |
| `-Q-D` | negative control | 53 | 0.503 | 0.925 | -0.421 | 2.886e-06 | 0.9210 | 0.9189 |

Delta is (arm − control). Negative delta = the ablation reduced the rate of exact-zero scoring.

### Correction across the 4 arm tests

| arm | raw p | Bonferroni | BH | survives BH? |
|---|---|---|---|---|
| `-Q-D` | 2.886e-06 | 1.155e-05 | 1.155e-05 | yes |
| `-Q` | 1.22e-05 | 4.878e-05 | 2.439e-05 | yes |
| `-D` | 0.03564 | 0.1425 | 0.04752 | yes |
| `-H` | 0.04156 | 0.1662 | 0.04156 | yes |

Note that `-H` and `-D` fail Bonferroni (0.166 and 0.143) and clear BH only marginally. `-Q` and `-Q-D` survive every correction by four orders of magnitude.

### Common subset — rows ablatable in ALL five arms

Arms above use different row subsets (n = 25 to 53), so their means are not directly comparable. Restricted to the **14** rows ablatable in every arm:

| arm | mean P(zero) | delta vs control | Wilcoxon p |
|---|---|---|---|
| `control` | 0.952 | — | — |
| `-H` | 0.881 | -0.071 | 0.593 |
| `-Q` | 0.667 | -0.286 | 0.04123 |
| `-D` | 0.810 | -0.143 | 0.2207 |
| `-Q-D` | 0.452 | -0.500 | 0.006557 |

In this clean factorial (n=14, small — treat as indicative) `-H` and `-D` are both null, while `-Q` and `-Q-D` retain their effect.

## Mean cosine per arm

If the effect runs through the flag rather than through similarity, cosine should be roughly flat across arms.

| arm | n | mean cosine |
|---|---|---|
| `control` | 56 | 0.9174 |
| `-Q` | 47 | 0.9215 |
| `-D` | 25 | 0.9226 |
| `-Q-D` | 53 | 0.9210 |
| `-H` | 38 | 0.8919 |

Spread across arms: **0.0307** (max 0.9226, min 0.8919).

Paired on the 14 rows present in every arm:

| arm | mean cosine | delta vs control |
|---|---|---|
| `control` | 0.9253 | +0.0000 |
| `-Q` | 0.9300 | +0.0046 |
| `-D` | 0.9227 | -0.0026 |
| `-Q-D` | 0.9253 | +0.0000 |
| `-H` | 0.8886 | -0.0367 |

**Cosine is flat for `-Q`, `-D` and `-Q-D` (all within ±0.005 of control), confirming those effects run through the flag and not through similarity. `-H` is the exception at −0.037** — it is the only arm that perturbed question generation, so part of its already-small effect may be a similarity artefact rather than a flag effect. `-H` therefore does not cleanly isolate the flag the way the other arms do.

## -H: did the ablation move the feature but not the flag?

Post-ablation `contrast_per_1k` and `empathy_per_1k` alongside P(zero), so a moved feature with an unmoved flag is distinguishable from a failed ablation.

| metric | control | -H | delta |
|---|---|---|---|
| `contrast_per_1k` | 1.189 | 0.265 | -0.924 |
| `empathy_per_1k` | 0.881 | 0.000 | -0.881 |
| `referral_per_1k` | 0.794 | 0.880 | +0.086 |
| `arm_len` | 891.368 | 821.921 | -69.447 |
| `p_zero` | 0.956 | 0.851 | -0.105 |
| `mean_cosine` | 0.911 | 0.892 | -0.019 |

Rows where -H strictly reduced `contrast_per_1k`: **27/38**
Rows where -H strictly reduced `empathy_per_1k`: **25/38**

Correlation between how much hedging was removed and how much P(zero) moved: **r = -0.014** (n=38).

## -H rows that flipped to non-zero

**5** of 38 ablatable -H rows moved from majority-zero under control to majority-non-zero under -H.

---

### Row 14

- control P(zero) **0.67** → -H P(zero) **0.00**
- control mean score 0.3077 → -H 0.8427
- contrast/1k 0.00 → 0.00; empathy/1k 1.17 → 0.00

**Question:** I have not exercised in years and just finished cancer treatment. Where do I even start?

**Control (original):**

> It's completely understandable to feel uncertain about starting exercise, especially after finishing cancer treatment. The important thing is that you're ready to take this step! 
> 
> Physical activity can help improve your energy levels, mood, and overall quality of life as you recover. Starting with small, manageable activities is key. Here are a few suggestions to get you started:
> 
> 1. **Walking**: Begin with short walks around your home or neighborhood. Even 5-10 minutes a day can be beneficial. 
> 
> 2. **Gentle Stretching**: Incorporate some gentle stretching or yoga to help improve flexibility and relaxation.
> 
> 3. **Bodyweight Exercises**: Simple exercises like seated leg lifts or wall push-ups can help you regain strength without requiring any equipment.
> 
> Would you like to explore a specific activity or set a goal for your physical activity?

**-H (empathy framing and contrast connectives removed):**

> The important thing is that you're ready to take this step! 
> 
> Physical activity can help improve your energy levels, mood, and overall quality of life as you recover. Starting with small, manageable activities is key. Here are a few suggestions to get you started:
> 
> 1. **Walking**: Begin with short walks around your home or neighborhood. Even 5-10 minutes a day can be beneficial. 
> 
> 2. **Gentle Stretching**: Incorporate some gentle stretching or yoga to help improve flexibility and relaxation.
> 
> 3. **Bodyweight Exercises**: Simple exercises like seated leg lifts or wall push-ups can help you regain strength without requiring any equipment.
> 
> Would you like to explore a specific activity or set a goal for your physical activity?

**Removed:**

- `It's completely understandable to feel uncertain about starting exercise, especially after finishing cancer treatment.`

---

### Row 42

- control P(zero) **0.67** → -H P(zero) **0.33**
- control mean score 0.3138 → -H 0.6077
- contrast/1k 0.72 → 0.00; empathy/1k 0.72 → 0.00

**Question:** I'm really concerned about how my prostate cancer treatments might affect my heart health. What can I do to stay healthy while managing my condition?

**Control (original):**

> It's completely understandable to be concerned about heart health while managing prostate cancer treatments. Many cancer treatments, especially hormone therapies, can increase the risk of cardiovascular issues. Here are some steps you can take to support your heart health:
> 
> 1. **Stay Active**: Regular physical activity can help manage weight, reduce fatigue, and improve overall health. Aim for at least 150 minutes of moderate aerobic activity each week, like walking or cycling. Gentle strength training can also be beneficial.
> 
> 2. **Eat a Heart-Healthy Diet**: Focus on a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats. Reducing sodium and saturated fats can help lower blood pressure and cholesterol levels.
> 
> 3. **Monitor Your Health Metrics**: Keep an eye on your blood pressure, blood sugar, and cholesterol levels. Regular check-ups with your healthcare team can help manage these metrics effectively.
> 
> 4. **Manage Stress**: Incorporate stress-reducing practices like meditation, yoga, or deep-breathing exercises into your routine. 
> 
> 5. **Stay Informed**: Keep communication open with your healthcare team about any concerns regarding your treatments and their effects on your heart health.
> 
> If you're interested, we can also look into specific actions you can take based on your current health metrics. Would you like to explore that?

**-H (empathy framing and contrast connectives removed):**

> Many cancer treatments, especially hormone therapies, can increase the risk of cardiovascular issues. Here are some steps you can take to support your heart health:
> 
> 1. **Stay Active**: Regular physical activity can help manage weight, reduce fatigue, and improve overall health. Aim for at least 150 minutes of moderate aerobic activity each week, like walking or cycling. Gentle strength training can also be beneficial.
> 
> 2. **Eat a Heart-Healthy Diet**: Focus on a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats. Reducing sodium and saturated fats can help lower blood pressure and cholesterol levels.
> 
> 3. **Monitor Your Health Metrics**: Keep an eye on your blood pressure, blood sugar, and cholesterol levels. Regular check-ups with your healthcare team can help manage these metrics effectively.
> 
> 4. **Manage Stress**: Incorporate stress-reducing practices like meditation, yoga, or deep-breathing exercises into your routine. 
> 
> 5. **Stay Informed**: Keep communication open with your healthcare team about any concerns regarding your treatments and their effects on your heart health.
> 
> If you're interested, we can also look into specific actions you can take based on your current health metrics. Would you like to explore that?

**Removed:**

- `It's completely understandable to be concerned about heart health while managing prostate cancer treatments.`

---

### Row 129

- control P(zero) **1.00** → -H P(zero) **0.33**
- control mean score 0.0000 → -H 0.6181
- contrast/1k 1.16 → 0.00; empathy/1k 1.16 → 0.00

**Question:** I am scared to do resistance training because of my lymphedema. What does the research say?

**Control (original):**

> It's completely understandable to feel apprehensive about resistance training with lymphedema. However, research shows that progressive resistance training can actually be safe and beneficial for those at risk of lymphedema, especially for breast cancer survivors. Studies indicate that engaging in a structured resistance program can help improve symptoms and even protect against lymphedema flare-ups when done with appropriate precautions, such as using a compression garment.
> 
> The key is to start slowly and focus on major muscle groups, ideally under the guidance of a fitness professional. Light weights or resistance bands can be effective, and exercises can be modified to suit your comfort level. 
> 
> If you're open to it, we can explore some gentle resistance training options that could help you feel more confident and supported. How does that sound?

**-H (empathy framing and contrast connectives removed):**

> Research shows that progressive resistance training can actually be safe and beneficial for those at risk of lymphedema, especially for breast cancer survivors. Studies indicate that engaging in a structured resistance program can help improve symptoms and even protect against lymphedema flare-ups when done with appropriate precautions, such as using a compression garment.
> 
> The key is to start slowly and focus on major muscle groups, ideally under the guidance of a fitness professional. Light weights or resistance bands can be effective, and exercises can be modified to suit your comfort level. 
> 
> If you're open to it, we can explore some gentle resistance training options that could help you feel more confident and supported. How does that sound?

**Removed:**

- `It's completely understandable to feel apprehensive about resistance training with lymphedema.`

---

### Row 143

- control P(zero) **1.00** → -H P(zero) **0.00**
- control mean score 0.0000 → -H 0.8873
- contrast/1k 1.30 → 0.00; empathy/1k 1.30 → 0.00

**Question:** I have prostate cancer and my doctor mentioned my PSA. Should I avoid intense exercise because it might affect my PSA levels?

**Control (original):**

> It's understandable to have concerns about how exercise might impact your PSA levels, especially when managing prostate cancer. However, research shows that exercise, including both resistance and aerobic activities, is generally safe for prostate cancer survivors and does not adversely affect PSA levels after consistent training.
> 
> In fact, engaging in regular physical activity can be beneficial for your overall health and may help mitigate some treatment-related side effects. It's always a good idea to discuss your specific exercise plans with your healthcare team, especially if you have any particular concerns or symptoms.
> 
> Getting back to your fitness journey, would you like to explore some safe and enjoyable exercise options that fit your current situation?

**-H (empathy framing and contrast connectives removed):**

> Research shows that exercise, including both resistance and aerobic activities, is generally safe for prostate cancer survivors and does not adversely affect PSA levels after consistent training.
> 
> In fact, engaging in regular physical activity can be beneficial for your overall health and may help mitigate some treatment-related side effects. It's always a good idea to discuss your specific exercise plans with your healthcare team, especially if you have any particular concerns or symptoms.
> 
> Getting back to your fitness journey, would you like to explore some safe and enjoyable exercise options that fit your current situation?

**Removed:**

- `It's understandable to have concerns about how exercise might impact your PSA levels, especially when managing prostate cancer.`

---

### Row 145

- control P(zero) **1.00** → -H P(zero) **0.00**
- control mean score 0.0000 → -H 0.8883
- contrast/1k 0.00 → 0.00; empathy/1k 1.22 → 0.00

**Question:** I am a postmenopausal woman and I am worried about breast cancer. Does exercise help?

**Control (original):**

> It's understandable to have concerns about breast cancer, especially as a postmenopausal woman. Research shows that regular exercise can significantly benefit your health and may help reduce the risk of breast cancer recurrence. Engaging in physical activity post-diagnosis has been linked to lower mortality rates from breast cancer and improved overall health.
> 
> Exercise can enhance your immune function, improve hormonal balance, and help manage weight, all of which are important factors in cancer survivorship. Activities like walking, strength training, and aerobic exercises can be effective. It's recommended to find a routine that you enjoy and can stick with, as consistency is key.
> 
> If you're interested, we can explore setting a specific exercise goal that fits your lifestyle. Would you like to discuss that?

**-H (empathy framing and contrast connectives removed):**

> Research shows that regular exercise can significantly benefit your health and may help reduce the risk of breast cancer recurrence. Engaging in physical activity post-diagnosis has been linked to lower mortality rates from breast cancer and improved overall health.
> 
> Exercise can enhance your immune function, improve hormonal balance, and help manage weight, all of which are important factors in cancer survivorship. Activities like walking, strength training, and aerobic exercises can be effective. It's recommended to find a routine that you enjoy and can stick with, as consistency is key.
> 
> If you're interested, we can explore setting a specific exercise goal that fits your lifestyle. Would you like to discuss that?

**Removed:**

- `It's understandable to have concerns about breast cancer, especially as a postmenopausal woman.`

## Unablatable rows (excluded from that arm's paired test)

| arm | unablatable | reason |
|---|---|---|
| `-H` | 18 | no empathy framing or removable contrast marker |
| `-Q` | 9 | no trailing question |
| `-D` | 31 | no referral sentence |
| `-Q-D` | 3 | no trailing question |
