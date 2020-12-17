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

def phi_intV(state, number_of_nodes, intVs): #特徴量にintVを組み込む
    #one-hotベクトル化する
    phi_s_int = np.zeros(number_of_nodes)
    for i in range(number_of_nodes):
        if i == state:
            phi_s_int[i] = intVs[i]
        else:
            phi_s_int[i] = 0 
    #行列で返す
    return phi_s_int

def Mu(traj, number_of_nodes, inintV, intVs):
    Mu_s = np.zeros(number_of_nodes)
    if inintV == 0:
        for s in traj:
            Mu_s = Mu_s + phi(s, number_of_nodes)
    else:
        for s in traj:
            Mu_s = Mu_s + phi_intV(s, number_of_nodes, intVs) 

    return Mu_s

def MuE(trajectories, number_of_nodes, inintV, intVs):
    MuE_m = np.zeros(number_of_nodes)
    
    for traj in trajectories:
        MuE_m += Mu(traj, number_of_nodes,inintV, intVs)
    
    MuE_m = MuE_m / len(trajectories)
    
    return MuE_m

def MaxEntIRL_graph(env, trajectories, delta, max_step, learning_rate, inintV, intVs):#MaxEnt本体 inintV:0ならば普通の逆強化学習を実行，それ以外はIntVを組み込む
    #P = np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)])
    P = env.P
    #x_size,y_size = env.shape[0],env.shape[1]
    #np.set_printoptions(threshold=np.inf)
    #print(P)
    
    global muE
    muE = MuE(trajectories, env.nS, inintV, intVs)
    #muE[4],muE[24] = 0.5,0.5
    print("muE")
    print(muE)
    #print(sum(muE))    
    theta = np.random.uniform(0, 1, size=env.nS)
    if inintV == 0:
        feature_matrix = np.eye(env.nS)
    else:
        feature_matrix = np.eye(env.nS)
        for i, intV in enumerate(intVs):
            feature_matrix[i][i] = intV
    
    print(feature_matrix)

    R = np.dot(theta, feature_matrix.T)    
    print("initial reward")
    print(R)    
    #for i in tqdm(range(n_epochs)):

    #float型の無限大を表す
    norm_grad = float('inf')
    st = 0
    while(norm_grad > delta):  
        #Rの計算

        R = np.dot(theta, feature_matrix.T)    
        R_max = np.max(R)
        #R = R / R_max

        """reward log
        if st % 5 == 0:
            print(R)
        st += 1
        """

        #インナーループ
        """Backward pass：後ろからたどる"""
        #nS:状態数　nA：行動数　状態数×行動数のゼロ行列
        policy = np.zeros([env.nS, max(env.nA)])

        Z_a = np.zeros([env.nS, max(env.nA)])
        """
        Z_a = [[]
                []]
        """
        
        #要素全てが1の行列
        Z_s = np.ones([env.nS])
        
        #Note:N回のイテレーションの”N”は，軌跡の長さ
        for n in range(max_step):
            
            Z_a = np.einsum("san, s, n -> sa", P, np.exp(R-R_max), Z_s) #nはnext_stateの意　アインシュタインの縮約記法
            Z_s = np.sum(Z_a, axis = 1) #Z_sの初期化位置は"ここ"
                        
        policy = np.einsum("sa, s -> sa", Z_a, 1/Z_s)#各状態における行動選択確率：：：これがsoft_Q_policy　P(a|s)
        
        """Forward pass"""
        Dt= np.zeros([max_step, env.nS]) #論文アルゴリズム中の Dを指す

        """
        Dt = [[0,0,0]
            [0,0,0,]]  例：状態数３，軌跡長２　その状態になる確率
        """
         
        #initialize mu[0] based on trajectories initial state
        if inintV == 0:
            for trajectory in trajectories:
                Dt[0][trajectory[0]] += 1
            Dt /= len(trajectories)
        else:
            for trajectory in trajectories:
                Dt[0][trajectory[0]] += 1 #intVs[trajectory[0]]
            Dt /= len(trajectories)
                
        for t in range(1, max_step):
            Dt[t] = np.einsum("s, sa, san -> n", Dt[t-1], policy, P)

        Ds = Dt.sum(axis=0)#頻度
        #print(Ds)
        
        #print(mu)
        #print(feature_matrix.T.dot(mu))
        #grad = muE - feature_matrix.T.dot(mu)
        
        #L2ノルム
        grad = muE - feature_matrix.T.dot(Ds)
        norm_grad = np.linalg.norm(grad, ord=2)

        #print(grad)
        #if np.random.rand() < 0.0001: 小数部と整数部を返す　ｆ：小数部
        f, i = math.modf(norm_grad*100)
        #print(f)
        if f < 0.001:
            print(norm_grad) 
        theta += learning_rate * grad #最大化問題なので勾配降下が勾配上昇（勾配を加える）になっていることに注意
        #print(theta)
                
    print("MaxEntIRL ended.")
    return R, policy, Ds
    
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

def make_scatter(x, y, inintV, correlation, ylb,xlb='Relative frequency'):
        #散布図の作成
        #figureの生成
    fig = plt.figure()
        #axをfigに設定
    ax = fig.add_subplot(1, 1, 1)
        #x軸：ノードを軌跡が通る回数の相対度数，y軸：報酬をsoftmaxで正規化したもの
    ax.scatter(x, y)
    plt.title("Correlation:{}".format(correlation))
    plt.xlabel(xlb)
    plt.ylabel(ylb)

    plt.grid(True)
        #散布図の表示
        #ax.set_ylim(bottom=0, top=0.001)
    plt.show()


        
############ ここまで関数 ########### ここからmain ##############

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx
    import graphenv
    import scipy.stats
    import shelve
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
    #エピソード数
    episode = 1

    #毎エピソードでの相関係数を格納
    correlation_list = []

    #データベースから保存してあるグラフを取り出す
    graph_data = shelve.open("database")



    #G = graphenv.make_random_graph(number_of_nodes, p) #ランダムな連結グラフの生成
    G = graph_data["graph_1"]
        #各ノードのintVを格納
    intVs = graphenv.spacesyntax(G)
        #各ノードのIntVをsoftmaxによって正規化したものを格納
    intVs_softmax = softmax(intVs)
    #intVを平均0，分散1に正規化
    intVs_standard = scipy.stats.zscore(intVs)

    #intVを0～1で正規化
    intVs_zero_one = (intVs - intVs.min()) / (intVs.max() - intVs.min())
           #グラフ環境の生成
    env = graphenv.Graphenv(G, reward)

    #報酬と頻度の相関を格納
    correlation_list_freq_reward = []
    #報酬とintの相関を格納
    correlation_list_intV_reward = []
    #頻度とintの相関を格納
    correlation_list_freq_intV = []


    for i in range(episode):

        #エキスパート軌跡の生成とtestdataの格納
        #startもgoalもランダムな軌跡
        expart_paths = graphenv.make_expart_paths(G, number_of_exparts)

        #ゴールは固定でスタートがランダムな軌跡
        #expart_paths = graphenv.make_random_goal_fixed_path(G, number_of_exparts, 39)
            #learndataの格納
        learn_data = []


        for i in range(number_of_exparts - num_testdata):
            learn_data.append(expart_paths.pop())

        #実験１　ランダムな軌跡から学習する場合
        #traj = learn_data
        path_list = [1, 30, 0, 7, 36]
        #path_list = [36, 38, 23, 26]
        #path_list = [36, 38, 23, 39, 22, 32, 31, 25, 11, 33, 19, 18, 20, 26]
        ##################################実験２####################################################
        #eq_traj = graphenv.make_one_expart_paths(G, number_of_exparts, goal=39, start=0)
        eq_traj = graphenv.make_one_expart_paths_any(path_list, number_of_exparts)
        traj = eq_traj #実験２　ある同一の軌跡から学習する場合
        ###########################################################################################

            
        gamma = 0.9
        #0.99,0.95,0.90,0.85,0.80
        #traj = learn_data #学習に使用するデータ 実験１

        num_traj = len(traj)
        max_step = len(traj[0])

            #軌跡の通ったノードの相対度数を生成
        rel_freq_data = graphenv.path_relative_frequency(expart_paths, number_of_nodes)

        #print(sum(rel_freq_data))

        #nx.draw(G, with_labels=True)
        #plt.show(G)

    
        print("intVs")
        print(intVs)   
        """reward estimation"""
            #delta は　勾配ベクトルにおけるL2ノルムの閾値（deltaを下回ったら推定完了とする）,learning_lateは勾配変化の学習率
        delta, learning_rate = 0.4, 0.02
        est_reward, soft_Q_policy, state_freq = MaxEntIRL_graph(env, traj, delta, max_step, learning_rate, inintV, intVs) 
        if inintV == 0:
            print("not in intV")
        else:
            print("in intV") 

            

            #報酬を保存
            #np.savetxt("R_X5Y5.csv",est_reward.reshape((X,Y)),delimiter=", ")

            #print(G)
            #print(softmax(np.array([-2,-56,-6,7,-4,3])))
        print("estimate reward")
        print(est_reward)
        print("####################################################################################")
        print("intVs")
        print(intVs) 

        print("action")
        print(env.nA)

        print("####################################################################################")
        print("state frequency")
        print(state_freq)
        print(state_freq.dtype)
    
        #相関係数の計算
        
        #correlation = est_correlation(rel_freq_data, est_reward)
        #print("correlation:{}".format(correlation))
        #correlation_list.append(correlation)
        
        #print("policy")
        #print(soft_Q_policy)
            
            #env_est = gridworld.GridWorld(grid_shape, est_reward)
            #est_agent = ValueIteration(env_est, gamma)
            
            #状態価値を出す#確率的な方策にも対応
            #V_est = est_agent.get_pi_value(soft_Q_policy)
            #print(V_est)
            #np.savetxt("V_Pro_1.csv",V_est.reshape((5,5)),delimiter=", ")
        
        """
        correlation_freq_reward = est_correlation(rel_freq_data, est_reward)
        correlation_intV_reward = est_correlation(intVs, est_reward)
        correlation_freq_intV = est_correlation(rel_freq_data, intVs)
        print("correlation_freq_reward")
        print(correlation_freq_reward)
        print("correlation_intV_reward")
        print(correlation_intV_reward)
        print("correlation_freq_intV")
        print(correlation_freq_intV)

        correlation_list_freq_reward.append(correlation_freq_reward)
        correlation_list_intV_reward.append(correlation_intV_reward)
        correlation_list_freq_intV.append(correlation_freq_intV)
        """
    """
    print("correlation_freq_reward")
    print(correlation_list_freq_reward)
    print("correlation_intV_reward")
    print(correlation_list_intV_reward)
    print("correlation_freq_intV")
    print(correlation_list_freq_intV)
    """

    
    """
    correlation_list_freq_reward_str = [str(n) for n in correlation_list_freq_reward]
    with open('correlation_list_freq_reward_inintV.txt', 'w') as f:
        f.write('\n'.join(correlation_list_freq_reward_str))
    
    correlation_list_intV_reward_str = [str(n) for n in correlation_list_intV_reward]
    with open('correlation_list_intV_reward_inintV.txt', 'w') as f:
        f.write('\n'.join(correlation_list_intV_reward_str))
    
    correlation_list_freq_intV_str = [str(n) for n in correlation_list_freq_intV]
    with open('correlation_list_freq_intV_inintV.txt', 'w') as f:
        f.write('\n'.join(correlation_list_freq_intV_str))
    """


    """
    if inintV == 0:
        make_scatter(rel_freq_data, est_reward, inintV, correlation_freq_reward, ylb="Rewards")

        make_scatter(intVs, est_reward, inintV, correlation_intV_reward, xlb="intV", ylb="Rewards")
    else:
        
        make_scatter(rel_freq_data, est_reward, inintV, correlation_freq_reward, ylb="Rewards(intV)")

        make_scatter(intVs, est_reward, inintV, correlation_intV_reward, xlb="intV", ylb="Rewards(intV)")

    make_scatter(rel_freq_data, intVs, inintV, correlation_freq_intV, ylb="intV")
    """



    
    est_reward_mean = np.zeros(number_of_nodes)
    for i in range(episode):
        est_reward_mean += est_reward
    est_reward_mean /= episode
    print("est_reward_mean")
    print(est_reward_mean)
    print('policy')
    print(soft_Q_policy)
    print(len(soft_Q_policy))

        #相関係数の計算

    #print("correlation:{}".format(correlation))

    #print(eq_traj[0])

    #print(avarage)

    #グラフの可視化
    """
    if inintV == 0:
        graphenv.graph_view_1st(G, est_reward, number_of_nodes, ylb="Reward")
    else:
        graphenv.graph_view_1st(G, est_reward, number_of_nodes, ylb="Reward(intV)")

    graphenv.graph_view_1st(G, rel_freq_data, number_of_nodes, ylb="frequency")
    graphenv.graph_view_1st(G, intVs, number_of_nodes, ylb="intV")
    
    """
    print("trajectory")
    print(eq_traj[0])

    if inintV == 0:
        graphenv.graph_view_2nd(G, eq_traj, est_reward_mean, number_of_nodes, ylb="Reward")
        graphenv.graph_view_2nd(G, eq_traj, state_freq, number_of_nodes, ylb="state frequency")
    else:
        graphenv.graph_view_2nd(G, eq_traj, est_reward_mean, number_of_nodes, ylb="Reward(intV)")
        graphenv.graph_view_2nd(G, eq_traj, state_freq, number_of_nodes, ylb="state frequency(intV)")
    graphenv.graph_view_2nd(G, eq_traj, intVs, number_of_nodes, ylb="intV")
    


    #correlation_list.append(correlation)
    
      
  
    

 
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

