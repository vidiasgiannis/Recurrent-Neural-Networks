import numpy as np
from rnn import *
from gru import GRU
from runner import Runner

# this will test the implementation of predict, acc_deltas, and acc_deltas_bptt in rnn.py, for a simple 3x2 RNN
y_exp = np.array([[ 0.39411072,  0.32179748,  0.2840918 ], [ 0.4075143,   0.32013043,  0.27235527], [ 0.41091755,  0.31606385,  0.2730186 ], [ 0.41098376,  0.31825833,  0.27075792], [ 0.41118931,  0.31812307,  0.27068762], [ 0.41356637,  0.31280332,  0.27363031], [ 0.41157736,  0.31584609,  0.27257655]])
s_exp = np.array([[ 0.66818777,  0.64565631], [ 0.80500806,  0.80655686], [ 0.85442692,  0.79322425], [ 0.84599959,  0.8270955 ], [ 0.84852462,  0.82794442], [ 0.89340731,  0.7811953 ], [ 0.86164528,  0.79916155], [ 0., 0.]])
U_exp = np.array([[ 0.89990596,  0.79983619], [ 0.5000714,   0.30009787]])
V_exp = np.array([[ 0.69787081,  0.30129314,  0.39888647], [ 0.60201076,  0.89866058,  0.70149262]])
W_exp = np.array([[ 0.57779081,  0.47890397], [ 0.22552931,  0.62294835], [ 0.39667988 , 0.19814768]])

loss_expected = 8.19118156763
loss2_expected = 3.29724981191
loss3_expected = 6.01420605985
mean_loss_expected = 1.16684249596
np_loss_expected = 0.887758278817

acc_expected = 1
acc1_np_lm_expected = 0
acc2_np_lm_expected = 1

# standard BP
deltaU_1_exp = np.array([[-0.11298744, -0.107331  ], [ 0.07341862, 0.06939134]])
deltaV_1_exp = np.array([[-0.06851441, -0.05931481, -0.05336094], [ 0.06079254,  0.0035937,   0.04875759]])
deltaW_1_exp = np.array([[-2.36320453, -2.24145091], [ 3.13861959,  2.93420307], [-0.77541506, -0.69275216]])

# BPPT
deltaU_3_exp = np.array([[-0.12007034, -0.1141893 ], [ 0.06377434, 0.06003115]])
deltaV_3_exp = np.array([[-0.07524721, -0.06495432, -0.05560471], [ 0.05465826, -0.00306904, 0.04567927]])
deltaW_3_exp = np.array([[-2.36320453, -2.24145091], [ 3.13861959,  2.93420307], [-0.77541506, -0.69275216]])

# binary prediction BP
deltaU_1_exp_np = np.array([[0.01926192, 0.01684262], [0.00719671, 0.0062928]])
deltaV_1_exp_np = np.array([[0., 0., 0.02156006], [0., 0., 0.00805535]])
deltaW_1_exp_np = np.array([[0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327 ]])

# binary prediction BPPT
deltaU_3_exp_np = np.array([[ 0.0216261, 0.01914693], [0.01044642, 0.00946145]])
deltaV_3_exp_np = np.array([[ 0.00223142, 0.00055566, 0.02156006], [0.00336126, 0.00046926, 0.00805535]])
deltaW_3_exp_np = np.array([[ 0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327]])

vocabsize = 3
hdim = 2
# RNN with vocab size 3 and 2 hidden layers
# Note that, for the binary prediction output vocab size should be 2 
# for test case simplicity, here we will use the same input and vocab size
r = RNN(vocabsize,hdim,vocabsize)
r.V[0][0]=0.7
r.V[0][1]=0.3
r.V[0][2]=0.4
r.V[1][0]=0.6
r.V[1][1]=0.9
r.V[1][2]=0.7

r.W[0][0]=0.6
r.W[0][1]=0.5
r.W[1][0]=0.2
r.W[1][1]=0.6
r.W[2][0]=0.4
r.W[2][1]=0.2

r.U[0][0]=0.9
r.U[0][1]=0.8
r.U[1][0]=0.5
r.U[1][1]=0.3


x = np.array([0,1,2,1,1,0,2])
d = np.array([1,2,1,1,1,1,1])
d_np = np.array([0])
d1_lm_np = np.array([2,0])
d2_lm_np = np.array([0,2])
x2 = np.array([1,1,0])
d2 = np.array([1,0,2])
x3 = np.array([1,1,2,1,2])
d3 = np.array([1,2,1,2,1])

p = Runner(r)

print("### predicting y")
y,s = r.predict(x)
if not np.isclose(y_exp, y, rtol=1e-08, atol=1e-08).all():
	print("y expected\n{0}".format(y_exp))
	print("y received\n{0}".format(y))
else:
	print("y passed")
if not np.isclose(s_exp, s, rtol=1e-08, atol=1e-08).all():
	print("\ns expected\n{0}".format(s_exp))
	print("s received\n{0}".format(s))
else:
	print("s passed")

print("\n### computing loss and mean loss")
loss = p.compute_loss(x,d)
loss2 = p.compute_loss(x2,d2)
loss3 = p.compute_loss(x3,d3)
mean_loss = p.compute_mean_loss([x,x2,x3],[d,d2,d3])
if not np.isclose(loss_expected, loss, rtol=1e-08, atol=1e-08) or not np.isclose(loss2_expected, loss2, rtol=1e-08, atol=1e-08) or not np.isclose(loss3_expected, loss3, rtol=1e-08, atol=1e-08):
	print("loss expected: {0}".format(loss_expected))
	print("loss received: {0}".format(loss))
	print("loss2 expected: {0}".format(loss2_expected))
	print("loss2 received: {0}".format(loss2))
	print("loss3 expected: {0}".format(loss3_expected))
	print("loss3 received: {0}".format(loss3))
else:
	print("loss passed")
if not np.isclose(mean_loss_expected, mean_loss, rtol=1e-08, atol=1e-08):
	print("mean loss expected: {0}".format(mean_loss_expected))
	print("mean loss received: {0}".format(mean_loss))
else:
	print("mean loss passed")


print("\n### standard BP")
r.acc_deltas(x,d,y,s)
if not np.isclose(deltaU_1_exp, r.deltaU).all():
	print("\ndeltaU expected\n{0}".format(deltaU_1_exp))
	print("deltaU received\n{0}".format(r.deltaU))
else:
	print("deltaU passed")
if not np.isclose(deltaV_1_exp, r.deltaV).all():
	print("\ndeltaV expected\n{0}".format(deltaV_1_exp))
	print("deltaV received\n{0}".format(r.deltaV))
else:
	print("deltaV passed")
if not np.isclose(deltaW_1_exp, r.deltaW).all():
	print("\ndeltaW expected\n{0}".format(deltaW_1_exp))
	print("deltaW received\n{0}".format(r.deltaW))
else:
	print("deltaW passed")

print("\n### BPTT with 3 steps")
r.deltaU.fill(0)
r.deltaV.fill(0)
r.deltaW.fill(0)

r.acc_deltas_bptt(x,d,y,s,3)
if not np.isclose(deltaU_3_exp, r.deltaU).all():
	print("\ndeltaU expected\n{0}".format(deltaU_3_exp))
	print("deltaU received\n{0}".format(r.deltaU))
else:
	print("deltaU passed")
if not np.isclose(deltaV_3_exp, r.deltaV).all():
	print("\ndeltaV expected\n{0}".format(deltaV_3_exp))
	print("deltaV received\n{0}".format(r.deltaV))
else:
	print("deltaV passed")
if not np.isclose(deltaW_3_exp, r.deltaW).all():
	print("\ndeltaW expected\n{0}".format(deltaW_3_exp))
	print("deltaW received\n{0}".format(r.deltaW))
else:
	print("deltaW passed")


# # BINARY PREDICTION TEST

print("\n### computing binary prediction loss")
np_loss = p.compute_loss_np(x,d_np)
if not np.isclose(np_loss_expected, np_loss, rtol=1e-08, atol=1e-08):
	print("np loss expected: {0}".format(np_loss_expected))
	print("np loss received: {0}".format(np_loss))
else:
	print("np loss passed")

print("\n### binary prediction BP")
r.deltaU.fill(0)
r.deltaV.fill(0)
r.deltaW.fill(0)

r.acc_deltas_np(x,d_np,y,s)
if not np.isclose(deltaU_1_exp_np, r.deltaU).all():
	print("\ndeltaU expected\n{0}".format(deltaU_1_exp_np))
	print("deltaU received\n{0}".format(r.deltaU))
else:
	print("deltaU passed")
if not np.isclose(deltaV_1_exp_np, r.deltaV).all():
	print("\ndeltaV expected\n{0}".format(deltaV_1_exp_np))
	print("deltaV received\n{0}".format(r.deltaV))
else:
	print("deltaV passed")
if not np.isclose(deltaW_1_exp_np, r.deltaW).all():
	print("\ndeltaW expected\n{0}".format(deltaW_1_exp_np))
	print("deltaW received\n{0}".format(r.deltaW))
else:
	print("deltaW passed")


print("\n### binary prediction BPTT with 3 steps")
r.deltaU.fill(0)
r.deltaV.fill(0)
r.deltaW.fill(0)

r.acc_deltas_bptt_np(x,d_np,y,s,3)
if not np.isclose(deltaU_3_exp_np, r.deltaU).all():
	print("\ndeltaU expected\n{0}".format(deltaU_3_exp_np))
	print("deltaU received\n{0}".format(r.deltaU))
else:
	print("deltaU passed")
if not np.isclose(deltaV_3_exp_np, r.deltaV).all():
	print("\ndeltaV expected\n{0}".format(deltaV_3_exp_np))
	print("deltaV received\n{0}".format(r.deltaV))
else:
	print("deltaV passed")
if not np.isclose(deltaW_3_exp_np, r.deltaW).all():
	print("\ndeltaW expected\n{0}".format(deltaW_3_exp_np))
	print("deltaW received\n{0}".format(r.deltaW))
else:
	print("deltaW passed")


print("\n### compute accuracy for binary prediction")
acc = p.compute_acc_np(x, d_np)
if acc != acc_expected:
	print("acc expected\n{0}".format(acc_expected))
	print("acc received\n{0}".format(acc))
else:
	print("acc passed")

# # GRU Test

r = GRU(vocabsize,hdim,vocabsize)
r.Vh[0][0]=0.7
r.Vh[0][1]=0.3
r.Vh[0][2]=0.4
r.Vh[1][0]=0.6
r.Vh[1][1]=0.9
r.Vh[1][2]=0.7

r.Uh[0][0]=0.6
r.Uh[0][1]=0.4
r.Uh[1][0]=0.3
r.Uh[1][1]=0.8

r.Vr[0][0]=0.2
r.Vr[0][1]=0.7
r.Vr[0][2]=0.1
r.Vr[1][0]=0.9
r.Vr[1][1]=0.6
r.Vr[1][2]=0.5

r.Ur[0][0]=0.1
r.Ur[0][1]=0.9
r.Ur[1][0]=0.4
r.Ur[1][1]=0.6

r.Vz[0][0]=0.6
r.Vz[0][1]=0.8
r.Vz[0][2]=0.9
r.Vz[1][0]=0.3
r.Vz[1][1]=0.2
r.Vz[1][2]=0.7

r.Uz[0][0]=0.9
r.Uz[0][1]=0.5
r.Uz[1][0]=0.9
r.Uz[1][1]=0.3

r.W[0][0]=0.6
r.W[0][1]=0.5
r.W[1][0]=0.2
r.W[1][1]=0.6
r.W[2][0]=0.4
r.W[2][1]=0.2

p = Runner(r)

y_exp = np.array([[0.3528942, 0.33141165, 0.31569415], [0.36113545, 0.33936667, 0.29949788], [0.36692804, 0.33951603, 0.29355593], [0.37231812, 0.3423382,  0.28534368], [0.37673929, 0.34350021, 0.2797605 ], [0.38331333, 0.33917522, 0.27751145], [0.38581534, 0.33818498, 0.27599968]])
s_exp = np.array([[0.21415391, 0.22854546], [0.26690114, 0.44588503], [0.32573672, 0.52650084], [0.36993061, 0.64022728], [0.41051185, 0.71839057], [0.49285591, 0.74806453], [0.52161192, 0.76878925], [0.,         0.        ]])

deltaUr_3_exp_np = [[0.00193795, 0.00314452], [0.00195132, 0.00320871]]
deltaVr_3_exp_np = [[0.00103786, 0.00190269, 0.00172727], [0.00109623, 0.0025203, 0.00126202]]
deltaUz_3_exp_np = [[-0.00942043, -0.01576331], [-0.00216407, -0.00358799]]
deltaVz_3_exp_np = [[-0.01069625, -0.00785168, -0.00464374], [-0.00094875, -0.00397553, -0.00080911]]
deltaUh_3_exp_np = [[0.01726854, 0.02951899], [0.00359722, 0.00635103]]
deltaVh_3_exp_np = [[0.01217558, 0.03100759, 0.01551138], [0.00432377, 0.00515261, 0.00275538]]
deltaW_GRU_3_exp_np = [[ 0.32036604, 0.47217857], [-0.17640132, -0.25999298], [-0.14396472, -0.21218559]]

y,s = r.predict(x)
r.deltaUr.fill(0)
r.deltaVr.fill(0)
r.deltaUz.fill(0)
r.deltaVz.fill(0)
r.deltaUh.fill(0)
r.deltaVh.fill(0)
r.deltaW.fill(0)

r.acc_deltas_bptt_np(x, d_np, y, s, 3)


print("\n### predicting y,s and deltas GRU")
if not np.isclose(y_exp, y, rtol=1e-08, atol=1e-08).all():
	print("y expected\n{0}".format(y_exp))
	print("y received\n{0}".format(y))
else:
	print("y passed")
if not np.isclose(s_exp, s, rtol=1e-08, atol=1e-08).all():
	print("\ns expected\n{0}".format(s_exp))
	print("s received\n{0}".format(s))
else:
	print("s passed")


print("\n### binary prediction GRU with 3 steps")
if not (np.isclose(deltaUr_3_exp_np, r.deltaUr).all() and \
        np.isclose(deltaVr_3_exp_np, r.deltaVr).all() and \
        np.isclose(deltaUz_3_exp_np, r.deltaUz).all() and \
        np.isclose(deltaVz_3_exp_np, r.deltaVz).all() and \
        np.isclose(deltaUh_3_exp_np, r.deltaUh).all() and \
        np.isclose(deltaVh_3_exp_np, r.deltaVh).all() and \
        np.isclose(deltaW_GRU_3_exp_np, r.deltaW).all()):
	print("    deltaUr expected\n{0}".format(deltaUr_3_exp_np))
	print("    deltaUr received\n{0}".format(r.deltaUr))
	print("    deltaVr expected\n{0}".format(deltaVr_3_exp_np))
	print("    deltaVr received\n{0}".format(r.deltaVr))
	print("    deltaUz expected\n{0}".format(deltaUz_3_exp_np))
	print("    deltaUz received\n{0}".format(r.deltaUz))
	print("    deltaVz expected\n{0}".format(deltaVz_3_exp_np))
	print("    deltaVz received\n{0}".format(r.deltaVz))
	print("    deltaUh expected\n{0}".format(deltaUh_3_exp_np))
	print("    deltaUh received\n{0}".format(r.deltaUh))
	print("    deltaVh expected\n{0}".format(deltaVh_3_exp_np))
	print("    deltaVh received\n{0}".format(r.deltaVh))
	print("    deltaW expected\n{0}".format(deltaW_GRU_3_exp_np))
	print("    deltaW received\n{0}".format(r.deltaW))
else:
	print("deltaUr passed")
	print("deltaVr passed")
	print("deltaUz passed")
	print("deltaVz passed")
	print("deltaUh passed")
	print("deltaVh passed")
	print("deltaW  passed")
