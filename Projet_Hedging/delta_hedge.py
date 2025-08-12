import numpy as np


def delta_hedge_calls(N,r,K,S,T):
    deltas = [0.522,0.458,0.400,0.596,0.693,0.774,0.771,0.706,0.674,0.787,0.550,
              0.413,0.542,0.591,0.768,0.759,0.865,0.978,0.990,1,1]
    cours_action = [49,48.12,47.37,50.25,51.75,53.12,53.00,51.87,51.38,53,49.88,48.5,49.88,50.37,52.13,
                    51.88,52.87,54.87,54.62,55.87,57.25]
    P = deltas[0]*N*S
    c_i = [P*(r/365)*7]
    c_a = [deltas[0]*N*S]
    A = [deltas[0]*N]
    c_c = [P]
    for k in range(1,len(deltas)):
        a = (deltas[k] - deltas[k-1])*N
        A.append(a)
        c_a.append(a*cours_action[k])
        c_c.append(c_c[k-1] + a*cours_action[k])
        c_i.append(c_c[k-1]*(r/365)*7)
        
        
    return (A,c_a,c_c,c_i)

print(delta_hedge_calls(100000,0.05,50,49,12))