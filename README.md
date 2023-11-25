# Coupled-Hydro-thermal-geoelectrical-parameter-estimator
A parameter optimization scheme to estimate the subsurface hydrological, thermal, and geophysical parameters by using soil moisture, temperature, and apparent resistivity observation data.

水文模擬參數最佳化（模型參數校驗）
參數估計流程圖：

 ![image](https://github.com/p56075607/Coupled-Hydro-thermal-geoelectrical-parameter-estimator/blob/main/Flow_chart.svg)
- ref：
    
    [Calibration and Uncertainty Analysis for Complex Environmental Models](https://www.notion.so/Calibration-and-Uncertainty-Analysis-for-Complex-Environmental-Models-ce728ec35f9944cca405e2249120d702?pvs=21) 
    
    [A python framework for environmental model uncertainty analysis](https://www.notion.so/A-python-framework-for-environmental-model-uncertainty-analysis-0f7d5fe28ded4ec1ae68a17f8c841569?pvs=21) 
    
1. 模擬參數最佳化目的
    - more
        
        引言：現實水文地質參數(k)非常複雜，而我們能取得的觀測量(h)往往非常有限、The null space
        
2. 逆推問題正則化與參數求解
    - more
        
        簡單介紹正則化：Manual、SVD decomposition、Tikhonov
        
        本研究不討論算術正則化相關問題，因為[Jafarov et al., 2020]提到利用單一權重定義的目標函式方式已能夠達到合理的參數估計結果
        
3. 參數估計軟體
4. 最佳化演算法
    - more
        
        最佳化演算法：Levenberg-Marquardt method
        
        1. ****Levenberg-Marquardt — Solving the nonlinear inverse problem in an iterative way****
            - LM 結合了Gradient descent 法(一階導數)與 Gauss-Newton 法(二階導數)的求解概念，調整lambda 以**穩定**迭代尋找目標函式最小值
                
                ![>](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>.png)
                
                [https://www.youtube.com/watch?v=2ToL9zUR8ZI](https://www.youtube.com/watch?v=2ToL9zUR8ZI)
                
                *注意：這裡的λ與tikhonov regularization 裡的 smooth factor λ 不同
                
            - PEST “Estimation mode”
                
                ![from PEST user manual1 p.63](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%201.png)
                
                from PEST user manual1 p.63
                
            - Non-linear parameter estimation
                
                
                ![X is the model operation matrix](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%202.png)
                
                X is the model operation matrix
                
                ![J is the Jacobian matrix during the iteration ](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%203.png)
                
                J is the Jacobian matrix during the iteration 
                
                ![等號右邊第一項代表二次微分的求解方向，第二項代表一次微分的求解方向。](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%204.png)
                
                等號右邊第一項代表二次微分的求解方向，第二項代表一次微分的求解方向。
                
                **針對 Marquardt lambda *“PEST implements a trial and error procedure that makes maximum use of spare computing capacity in a parallel environment.”***
                
            
            - Marquardt lambda
                
                ![>](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Estimation%20of%20subsurface%20porosities%20and%20thermal%20co%20f17ebe41630c4567b29676a4aab1920f/>.png)
                
                if lambda is very high, becomes…
                
                ![399C085E-A65C-4C54-95BB-7C518622B744.jpeg](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Estimation%20of%20subsurface%20porosities%20and%20thermal%20co%20f17ebe41630c4567b29676a4aab1920f/399C085E-A65C-4C54-95BB-7C518622B744.jpeg)
                
                The Marquardt lambda start from large and decrease as we approach the obj. func. minimum to avoid hemstitching
                
                ![>](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Estimation%20of%20subsurface%20porosities%20and%20thermal%20co%20f17ebe41630c4567b29676a4aab1920f/>%201.png)
                
                ***PEST implements a trial and error procedure that makes maximum use of spare computing capacity in a parallel environment in determining Marquardt lambda.***
                
            - ****Marquardt lambda 的意義**** (from PEST book p.83)
                1. ****Marquardt lambda 可視為移動步長：****當 lambda 很大的時候，表示傾向往一階導數也就是梯度方向求解，在這個情況下求解移動的步長會較牛頓法短，反之亦然。
                2. ****Marquardt lambda 也可視為一種正則化手段：****當求最小目標函式過程中遇到無法計算反矩陣時，適當於矩陣的主對角線上加上一正數，使得其反矩陣存在，增加逆推過程中的穩定性。
            - 最佳移動步長的依據
                
                每次迭代中，沿著 Levenberg-Marquardt 所指示的求解方向，找尋能夠使目標函式出現最小值的位置，此移動長度視為最佳移動步長
                
                ![沿著求解方向將目標函式對 alpha 作微分並找尋=0時的解，即為目標函式的最小值步長](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%205.png)
                
                沿著求解方向將目標函式對 alpha 作微分並找尋=0時的解，即為目標函式的最小值步長
                
            - 設定參數上下界的原因
                1. 避免出現不符物理規則之參數值，以確保執行模擬過程中不會出錯
                2. 避免出現模擬者認為之不合理的參數條件，此項的參考可能依據現地地質先備資訊、觀測資料或是在進行參數校驗最佳化前的初步模擬結果中得到
        
        收斂條件：最大迭代次數、目標函式相對減少量、參數相對改變量
        
        
5. 參數估計**不確定性分析**
    - more
        
        非線性不確定性分析FOSM-based：Bayes equation、Schur's complement
        
        - Bayes equation
            
            ![>](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%206.png)
            
            假設：
            
            1. 模型參數與輸出資料間關係為線性(Jacobian)、
            2. 先、後驗參數不確定性之機率密度分佈皆為*[multivariate*-gaussian](https://www.google.com/search?sxsrf=AJOqlzVjV_yjyRCx0kj0vF-s_EgRP8JHVA:1679360802735&q=multivariate-gaussian&sa=X&ved=2ahUKEwit1eLA6uv9AhWjT2wGHQzaA50Q7xYoAHoECAkQAQ)
            3. 測量誤差之雜訊也為 Gaussian 分布
            - Schur's complement conditional uncertainty propagation
                
                ![截圖 2023-03-18 上午9.22.14.png](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/%25E6%2588%25AA%25E5%259C%2596_2023-03-18_%25E4%25B8%258A%25E5%258D%25889.22.14.png)
                
                - Posterior mean and covariance matrix from conditioning equations
                    
                    現在我們關心的是參數估計的不確定性，也就是經過測量資料的calibration後的參數後驗分布，也就是$f(\bf{k}|\bf{h})$ 
                    
                    *“the posterior probability distribuation of the calibrated model parameters **k** conditoned on the observations **h**”*
                    
                    ![在假設**模型參數誤差**與**測量誤差**彼此互相獨立情形下](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/%25E6%2588%25AA%25E5%259C%2596_2023-03-18_%25E4%25B8%258A%25E5%258D%25889.30.51.png)
                    
                    在假設**模型參數誤差**與**測量誤差**彼此互相獨立情形下
                    
                    套用條件機率以及誤差傳遞的概念
                    
                    - Conditional posterior mean
                        
                        ![截圖 2023-03-18 上午9.31.59.png](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/%25E6%2588%25AA%25E5%259C%2596_2023-03-18_%25E4%25B8%258A%25E5%258D%25889.31.59.png)
                        
                    - Conditional posterior covariance：**Schur's complement**
                        
                        ![截圖 2023-03-18 上午9.32.29.png](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/%25E6%2588%25AA%25E5%259C%2596_2023-03-18_%25E4%25B8%258A%25E5%258D%25889.32.29.png)
                        
            - 實作上的假設
                
                ![>](Reference%20Journal%20Articals%20f12ccb486cef4c778ffc2a157244405b/Calibration%20and%20Uncertainty%20Analysis%20for%20Complex%20E%20ce728ec35f9944cca405e2249120d702/>%207.png)
                
                - **Measurement noise covariance $C(\varepsilon)$：**
                    
                    When implemented by PESTPP-GLM, an additional assumption is made. It is that the **standard deviation of measurement noise** associated with each observation **is proportional current observation residual**. This attempts to account for how well (or otherwise) the model reproduces the observations. **If the model is not fitting a given observation, then that implies a large uncertainty for that observation**, which in turn prevents the observation from conditioning the parameter(s) it is sensitive to. Note, the residual weight adjustment process will never increase weights (measurement noise standard deviations will never be decreased).
                    
                    The PESTPP-GLM posterior parameter (and optional forecast) uncertainty analyses require an **observation noise covariance matrix**. As presently coded, this matrix is formed **from the observation weights** (e.g., it is diagonal). However, these weights assume that the final measurement objective function is equal to the number of non-zero weighted observations – this almost never happens, largely as a result of model error. This is a problem for posterior FOSM-based uncertainty analyses because the weights in the control imply a more complete transfer of information from observations to parameters than was actually achieved. To rectify this issue, **PESTPP-GLM will scale the weights used for posterior FOSM analysis to account for the final residuals**, adopting Morozov’s discrepancy principal. The scaled weights are written to separate residuals file for inspection named case.fosm_reweight.rei.
                    
                    → 在實作時程式中假設測量誤差的共變異矩陣為一個對角線矩陣，意旨各測量值間皆彼此獨立，而主對角線上之值，則設為當次迭代的觀測殘差乘以該資料的權重，此定義之理由為當模擬值與測量值之間的殘差較大時，代表兩者擬合度較差，也意味著此觀測值中可能參雜較大的不確定性。
                    
                - **Prior parameter covariance $C(\textbf{k})$：**
                    
                    Unless a parcov() control variable is provided in the PEST control file, **PESTPP-GLM assumes that all adjustable parameters are statistically independent**. In this case, by default, the **prior standard deviation of each parameter is calculated as a quarter of the difference between its upper and lower bounds** as provided in the PEST control file. However, the par_sigma_range() control variable (the default value for which is 4.0) can be used to specify that the difference between parameter bounds is equivalent to a different number of standard deviations from this. If a parameter is log-transformed, the prior standard deviation of its log is calculated from the difference between the logs of its bounds.
                    
                    → 而先驗參數共變異矩陣則也是假設為一個對角矩陣，也就是假設每個調整參數間也是呈統計上獨立，並且根據軟體預設算法計算先驗參數共變異矩陣，其計算方式為將參數先驗標準偏差設為參數估計上下界差異的四分之一，如此一來參數估計的後驗共變異矩陣可藉由以上定義之測量誤差共變異矩陣及先驗參數共變異矩陣，以及每次迭代運算之亞可比矩陣計算出。

# Acknowledgements
感謝本研究的執行計畫:發展及應用二維地電阻層析成像技術推 估農地之土壤水文特性，及行政院農業委員會農業試驗所對於本研究的支持
