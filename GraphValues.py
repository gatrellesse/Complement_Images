import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('/home/gatrellesse/Complement_Images/Microbit_Values.xlsx', sheet_name='Plan1')

# Plot
plt.figure(figsize=(10, 4))
plt.plot(df.index, df, label="ADC value (mV)")
plt.xlabel("Sample Number")
plt.ylabel("ADC Reading (mV)")
plt.title("ADC Signal over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig("fall_detection_plot.png")  # Save as PNG
# plt.savefig("fall_detection_plot.pdf")  # Or PDF
plt.close() 