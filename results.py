import matplotlib.pyplot as plt 

names = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", \
        "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", \
        "1.7", "1.8", "1.9"]

l = len(names)
file_dir = "./Results/"
psnr_clean = []
psnr_degraded = []

for i in range(l):
    filename = file_dir + names[i] + ".txt"
    with open(filename, "r") as f:
        r = f.read()
        r = r.split(' ')
        r = [float(x) for x in r]
        psnr_clean.append(r[0])
        print(r[0])
        psnr_degraded.append(r[1])
        

plt.plot(psnr_clean, 'ro')
plt.plot(psnr_degraded)
plt.axis([0.1, 1.9, 0, 35])
plt.show()


