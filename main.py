import matplotlib.pyplot as plt

print("--- F1 FRONT WING SIMULATOR ---")

wing_area = float(input("Enter Front Wing Area (m^2): "))
cl = float(input("Enter Lift Coefficient: "))
stress_factor = 2.5
material_strength = 500000

speeds = [50, 100, 150, 200, 250, 300]

downforces = []
safety_factors = []

print("\n--- RESULTS ---")
print("Speed \t Downforce \t Stress \t Safety Factor")

for kmh in speeds:
    v = kmh / 3.6

    downforce = 0.5 * 1.225 * (v ** 2) * cl * wing_area
    stress = downforce * stress_factor
    sf = material_strength / stress

    downforces.append(downforce)
    safety_factors.append(sf)

    print(kmh, "\t", downforce, "\t", stress, "\t", sf)

plt.plot(speeds, downforces, 'b-o', label='Downforce')
plt.plot(speeds, safety_factors, 'r-s', label='Safety Factor')

plt.xlabel('Speed (km/h)')
plt.ylabel('Value')
plt.title('F1 Front Wing')
plt.legend()
plt.grid(True)

plt.show()

print("done")