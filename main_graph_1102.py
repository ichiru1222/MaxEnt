# -*- coding: utf-8 -*-
"""
MaxEntIRL

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from statistics import mean, median,variance,stdev
import math

######################MAXENT#########################################################
### Φ(s)の計算 ######################################
def phi(state, number_of_nodes): #特徴量
    #one-hotベクトル化する
    phi_s = np.zeros(number_of_nodes)
    for i in range(number_of_nodes):
        if i == state:
            phi_s[i] = 1
        else:
            phi_s[i] = 0 
    #行列で返す
    return phi_s 

def phi_intV(state, number_of_nodes): #特徴量にintVを組み込む
    #one-hotベクトル化する
    phi_s = np.zeros(number_of_nodes)
    for i in range(number_of_nodes):
        if i == state:
            phi_s[i] = intVs_softmax[i]
        else:
            phi_s[i] = 0 
    #行列で返す
    return phi_s 

def Mu(traj, number_of_nodes, inintV):
    Mu_s = np.zeros(number_of_nodes)
    if inintV == 0:
        for s in traj:
            Mu_s = Mu_s + phi(s, number_of_nodes)
    else:
        for s in traj:
            Mu_s = Mu_s + phi_intV(s, number_of_nodes) 

    return Mu_s

def MuE(trajectories, number_of_nodes, inintV):
    MuE_m = np.zeros(number_of_nodes)
    
    for traj in trajectories:
        MuE_m += Mu(traj, number_of_nodes,inintV)
    
    MuE_m = MuE_m / len(trajectories)
    
    return MuE_m

def MaxEntIRL_graph(env, trajectories, delta, max_step, learning_rate, inintV):#MaxEnt本体 inintV:0ならば普通の逆強化学習を実行，それ以外はIntVを組み込む
    #P = np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)])
    P = env.P
    #x_size,y_size = env.shape[0],env.shape[1]
    
    global muE
    muE = MuE(trajectories, env.nS, inintV)
    #muE[4],muE[24] = 0.5,0.5
    #print(muE)
    #print(sum(muE))    
    theta = np.random.uniform(-0.5, 0.5, size=env.nS)
    feature_matrix = np.eye(env.nS)

    R = np.dot(theta, feature_matrix.T)    
    print("initial reward")
    print(R)    
    #for i in tqdm(range(n_epochs)):

    #float型の無限大を表す
    norm_grad = float('inf')

    while(norm_grad > delta):  
        #Rの計算
        R = np.dot(theta, feature_matrix.T)    

        #インナーループ
        """Backward pass：後ろからたどる"""
        #nS:状態数　nA：行動数　状態数×行動数のゼロ行列
        policy = np.zeros([env.nS, max(env.nA)])

        Z_a = np.zeros([env.nS, max(env.nA)])
        
        #要素全てが1の行列
        Z_s = np.ones([env.nS])
        
        #Note:N回のイテレーションの”N”は，軌跡の長さ
        for n in range(max_step):            
            Z_a = np.einsum("san, s, n -> sa", P, np.exp(R), Z_s) #nはnext_stateの意　アインシュタインの縮約記法
            Z_s = np.sum(Z_a, axis = 1) #Z_sの初期化位置は"ここ"
                        
        policy = np.einsum("sa, s -> sa", Z_a, 1/Z_s)#各状態における行動選択確率：：：これがsoft_Q_policy
        
        """Forward pass"""
        Dt = np.zeros([max_step, env.nS]) #論文アルゴリズム中の Dを指す
         
        #initialize mu[0] based on trajectories initial state
        for trajectory in trajectories:
            Dt[0][trajectory[0]] += 1
        Dt /= len(trajectories)
                
        for t in range(1, max_step):
            Dt[t] = np.einsum("s, sa, san -> n", Dt[t-1], policy, P) 
        Ds = Dt.sum(axis=0)#頻度
        
        #print(mu)
        #print(feature_matrix.T.dot(mu))
        #grad = muE - feature_matrix.T.dot(mu)
        
        #L2ノルム
        grad = muE - feature_matrix.T.dot(Ds)
        norm_grad = np.linalg.norm(grad, ord=2)

        #print(grad)
        #if np.random.rand() < 0.0001:
        f, i = math.modf(norm_grad*100)
        #print(f)
        if f < 0.001:
            print(norm_grad) 
        theta += learning_rate * grad #最大化問題なので勾配降下が勾配上昇（勾配を加える）になっていることに注意
        #print(theta)
                
    print("MaxEntIRL ended.")
    return R, policy
    
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color

def miyasui_plot(for_plot,X,Y):#Q値を可視化
    max_bool = for_plot == np.max(for_plot, axis=0)
    max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
    #max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')
    for_plot_array_round = np.round(for_plot, decimals=2)
    # 行動価値関数を表示
    ax = plt.gca()
    plt.xlim(0,X)
    plt.ylim(0,Y)
    ax.xaxis.set_ticklabels([])#x軸の数字を消す
    ax.yaxis.set_ticklabels([])#y軸の数字を消す
    
    for i in range(X):
        for j in range(Y):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            #rect2 = plt.Rectangle(xy =(i+0.37,j+0.4) , width=0.25,height=0.2,edgecolor='black',facecolor='white',alpha=1)
            ax.add_patch(rect)
            #ax.add_patch(rect2)
            # diag
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j,j+1),color='k',linewidth=.5)
            ax.add_line(diag)
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j+1,j),color='k',linewidth=.5)
            ax.add_line(diag)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # text
            plt.text(i+ 0.75, j+0.45, "%s" % (str(for_plot_array_round[0,x,y])), color=max_color_k[0,x,y])
            plt.text(i+ 0.4, j+0.8, "%s" % (str(for_plot_array_round[1,x,y])), color=max_color_k[1,x,y])
            plt.text(i+ 0.025, j+0.45, "%s" % (str(for_plot_array_round[2,x,y])), color=max_color_k[2,x,y])
            plt.text(i+ 0.4, j+0.1, "%s" % (str(for_plot_array_round[3,x,y])), color=max_color_k[3,x,y])
            plt.text(i+ 0.4, j+0.45, "%s" % (str(for_plot_array_round[4,x,y])), color=max_color_k[4,x,y])
    plt.show()

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def softmax(a):
    a_max = max(a)
    x = np.exp(a-a_max)
    u = np.sum(x)
    return x/u

def est_correlation(x, y):
        #相関係数の計算
    coref = np.corrcoef(x, y)
    correlation = coref[0][1]
    return correlation

def make_scatter(x, y, inintV, correlation):
        #散布図の作成
        #figureの生成
    fig = plt.figure()
        #axをfigに設定
    ax = fig.add_subplot(1, 1, 1)
        #x軸：ノードを軌跡が通る回数の相対度数，y軸：報酬をsoftmaxで正規化したもの
    ax.scatter(x, y)
    plt.title("Correlation:{}".format(correlation))
    plt.xlabel("Relative frequency")
    if inintV == 0:
        plt.ylabel("Rewards")
    else:
        plt.ylabel("Rewards(intV)")
    plt.grid(True)
        #散布図の表示
        #ax.set_ylim(bottom=0, top=0.001)
    plt.show()

        
############ ここまで関数 ########### ここからmain ##############

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import graphenv
 
    from value_iteration import ValueIteration


    
    #setting env
    #X,Y = 5,5
    #setting reward
    #grid_shape =[X,Y]
    #reward = np.full(np.prod(grid_shape), 0.0)
    #setting expert 

    #ノードの数
    number_of_nodes = 40
    #ノード間のエッジの生成確率
    p = 0.05
    #報酬をノードの数ごとに格納
    reward = np.zeros(number_of_nodes)
    #エキスパート軌跡の総数
    number_of_exparts = 500
    #intVを組み入れるかどうか，inintV=0 ならそのまま逆強化学習を実行，それ以外ならintVを組み込む
    inintV = 1
    #testdataの数
    num_testdata = 100
    #learndataの数
    num_learndata = number_of_exparts - num_testdata

    correlation_list = []

    G = graphenv.make_random_graph(number_of_nodes, p) #ランダムな連結グラフの生成
        #各ノードのintVを格納
    intVs = graphenv.spacesyntax(G)
        #各ノードのIntVをsoftmaxによって正規化したものを格納
    intVs_softmax = softmax(intVs)
        #グラフ環境の生成
    env = graphenv.Graphenv(G, reward)
        #エキスパート軌跡の生成とtestdataの格納
    expart_paths = graphenv.make_expart_paths(G, number_of_exparts)

        #learndataの格納
    learn_data = []


    for i in range(number_of_exparts - num_testdata):
        learn_data.append(expart_paths.pop())



    ##################################実験２####################################################
    eq_traj = graphenv.make_one_expart_paths(G, number_of_exparts)



        
    gamma = 0.9
        #0.99,0.95,0.90,0.85,0.80
    #traj =[[20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 8, 3, 4], [20, 15, 16, 11, 6, 7, 8, 9, 4], [20, 21, 16, 11, 6, 7, 8, 9, 4], [20, 15, 16, 17, 12, 13, 14, 9, 4], [20, 21, 22, 17, 12, 7, 8, 3, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 3, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 21, 16, 11, 12, 7, 2, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 22, 17, 18, 13, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 15, 10, 11, 12, 7, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 11, 12, 7, 2, 3, 4], [20, 21, 22, 23, 18, 13, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 9, 4], [20, 15, 10, 11, 12, 13, 8, 9, 4], [20, 15, 16, 17, 18, 13, 14, 9, 4], [20, 21, 22, 17, 12, 13, 8, 3, 4], [20, 15, 16, 17, 18, 13, 8, 3, 4], [20, 21, 22, 17, 12, 7, 8, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 11, 12, 13, 8, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 21, 22, 17, 18, 13, 14, 9, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 15, 16, 11, 12, 13, 8, 3, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 21, 22, 17, 12, 13, 8, 3, 4], [20, 15, 16, 17, 18, 13, 8, 9, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 15, 10, 11, 12, 7, 8, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 10, 11, 6, 1, 2, 3, 4], [20, 15, 10, 5, 6, 7, 8, 3, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 21, 22, 23, 18, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 17, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 12, 13, 14, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 23, 18, 13, 8, 9, 4], [20, 21, 16, 17, 12, 13, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 11, 6, 7, 8, 3, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 18, 13, 8, 9, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 11, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 7, 8, 9, 4], [20, 15, 16, 17, 18, 13, 8, 3, 4], [20, 21, 16, 11, 12, 7, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 12, 13, 8, 9, 4], [20, 21, 22, 17, 12, 7, 8, 9, 4], [20, 21, 22, 17, 18, 13, 8, 3, 4], [20, 21, 16, 11, 12, 7, 8, 9, 4], [20, 21, 22, 17, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 17, 18, 13, 8, 9, 4], [20, 21, 22, 17, 12, 7, 8, 9, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4]] 
    traj = learn_data #学習に使用するデータ 実験１

    #traj = eq_traj #実験２


    num_traj = len(traj)
    max_step = len(traj[0])

        #軌跡の通ったノードの相対度数を生成
    rel_freq_data = graphenv.path_relative_frequency(expart_paths, number_of_nodes)
        
    """reward estimation"""
        #delta は　勾配ベクトルにおけるL2ノルムの閾値（deltaを下回ったら推定完了とする）,learning_lateは勾配変化の学習率
    delta, learning_rate = 0.01, 0.015
    est_reward, soft_Q_policy = MaxEntIRL_graph(env, traj, delta, max_step, learning_rate, inintV) 
    if inintV == 0:
        print("not in intV")
    else:
        print("in intV") 

        #報酬をsoftmaxで正規化
    softmax_est_reward = softmax(est_reward)

        #報酬を保存
        #np.savetxt("R_X5Y5.csv",est_reward.reshape((X,Y)),delimiter=", ")

        #print(G)
        #print(softmax(np.array([-2,-56,-6,7,-4,3])))
    print("estimate reward")
    print(est_reward)

    print("####################################################################################")

    print("softmax reward")
    print(softmax_est_reward)
    print(sum(softmax_est_reward))
    #print(sum(softmax(est_reward)))
    #print(soft_Q_policy)
        
        #env_est = gridworld.GridWorld(grid_shape, est_reward)
        #est_agent = ValueIteration(env_est, gamma)
        
        #状態価値を出す#確率的な方策にも対応
        #V_est = est_agent.get_pi_value(soft_Q_policy)
        #print(V_est)
        #np.savetxt("V_Pro_1.csv",V_est.reshape((5,5)),delimiter=", ")
    print("####################################################################################")



        #相関係数の計算
    #coref = np.corrcoef(rel_freq_data, est_reward)
    #correlation = coref[0][1]
    #print("correlation:{}".format(correlation))

    #print(eq_traj[0])

    #print(avarage)

    #グラフの可視化
    #graphenv.graph_view(G, eq_traj, est_reward, number_of_nodes, ylb="Reward(intV)")
    #graphenv.graph_view(G, eq_traj, intVs_softmax, number_of_nodes, ylb="intV")
    


    #correlation_list.append(correlation)
"""
        #散布図の作成
        #figureの生成
    fig = plt.figure()
        #axをfigに設定
    ax = fig.add_subplot(1, 1, 1)
        #x軸：ノードを軌跡が通る回数の相対度数，y軸：報酬をsoftmaxで正規化したもの
    ax.scatter(rel_freq_data, est_reward)
    plt.title("Correlation:{}".format(correlation))
    plt.xlabel("Relative frequency")
    if inintV == 0:
        plt.ylabel("Rewards")
    else:
        plt.ylabel("Rewards(intV)")
    plt.grid(True)
        #散布図の表示
        #ax.set_ylim(bottom=0, top=0.001)
    plt.show()
"""
    #print(correlation_list)
    

 
"""

 ###############################miyasui_plot#################################################
    for_plot_a =[]
    #direction = {'stay':1, 'migi':2, 'hidari':3, 'sita':4, 'ue':5}
    stay, migi, hidari, sita, ue =[],[],[],[],[]
    list_k = soft_Q_policy.reshape((1,125)) #最適方策
    for i in range(X*Y):
        stay.append(list_k[0][5*(i+1)-1])
    for i in range(X*Y):
        migi.append(list_k[0][5*(i+1)-2])
    for i in range(X*Y):
        hidari.append(list_k[0][5*(i+1)-3])
    for i in range(X*Y):
        sita.append(list_k[0][5*(i+1)-4])
    for i in range(X*Y):
        ue.append(list_k[0][5*(i+1)-5])
                
    migi_s, ue_s, hidari_s, sita_s, stay_s = list(split_list(migi,X)), list(split_list(ue,X)) , list(split_list(hidari,X)), list(split_list(sita,X)), list(split_list(stay,X))
    migi_all,ue_all,hidari_all,sita_all,stay_all = [],[],[],[],[]
    
    for i in range(X):
        migi_all.append(migi_s[i])
        hidari_all.append(hidari_s[i])
        ue_all.append(ue_s[i])
        sita_all.append(sita_s[i])
        stay_all.append(stay_s[i])
    
    for_plot_a.append(migi_all)
    for_plot_a.append(ue_all)
    for_plot_a.append(hidari_all)
    for_plot_a.append(sita_all)
    for_plot_a.append(stay_all)
    
    plt.figure()
    miyasui_plot(for_plot_a,X,Y)
 """ 

########################################################################################

