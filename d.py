import matplotlib.pyplot as plt

# Plot a simple bar chart for outbreaks over time
outbreak_counts = df.groupby('year')['outbreak'].sum()  # Total outbreaks per year

plt.figure(figsize=(10, 6))
outbreak_counts.plot(kind='bar', color='red', alpha=0.7)
plt.title('Disease Outbreaks per Year')
plt.xlabel('Year')
plt.ylabel('Number of Outbreaks')
plt.show()
