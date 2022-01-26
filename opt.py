
def smash_log(x, B=10, d=0):
    return 1 / (1 + np.exp(-x * B)) - d

def j(z, xp, xn, m):
    z = np.round(z)
    sens = np.sum(np.dot(xp, z) >= m, dtype=np.uint8) / xp.shape[0]
    fpr = np.sum(np.dot(xn, z) >= m, dtype=np.uint8) / xn.shape[0]
    spec = 1 - fpr
    print(sens, spec)
    return sens + spec - 1

def j_exp(z, xp, xn, m, a=1, b=1):
    tpr = smash_log(np.dot(xp, z) - (m - 1)).sum() / xp.shape[0]
    fpr = smash_log(np.dot(xn, z) - (m - 1)).sum() / xn.shape[0]
    return -1 * (a*tpr - b*fpr)

bnds = ((0, 1),) * 16
init = np.random.choice([0, 1], 16)

m = 1
n = 5
con = sp.optimize.LinearConstraint(np.ones(16), m, n)

opt = sp.optimize.minimize(
    fun=j_exp,
    x0=init,
    args=(xp, xn, m),
    constraints=con,
    bounds=bnds
)

good = opt.x.round()
good_s = [symptom_list[i] for i in  np.where(good == 1)[0]]
good_s
j(good, xp, xn, m)

