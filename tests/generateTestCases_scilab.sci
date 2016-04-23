A = [1,2;,0,3] 
B2 = [0;1]

C1 = [1,0;0,1;0,0]
D12 = [0;0;1]
// Q = [1,0;0,1]
// R = [1]

Vx = 0.1*[1,0;0,1]

system = syslin('c',A,B2,C1,D12)

[K2, X2] = lqr(system)
[Kinf, Xinf, err] = leqr(system, Vx)

disp(K2)
disp(Kinf)
disp(err)

