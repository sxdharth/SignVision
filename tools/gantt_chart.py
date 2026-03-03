import pandas as pd
import matplotlib.pyplot as plt

# Define the project phases and their relative timing (in generic time units, e.g., days or weeks)
# We structure this so phases flow sequentially, with some overlaps if needed.
data = [
    # Phase 1: Setup
    {"Task": "Environment Setup", "Start": 0, "Duration": 2, "Phase": "Phase 1"},
    
    # Phase 2: Data Collection
    {"Task": "WLASL Dataset Processing", "Start": 2, "Duration": 4, "Phase": "Phase 2"},
    {"Task": "Custom Data Recording", "Start": 3, "Duration": 5, "Phase": "Phase 2"},
    
    # Phase 3: Data Processing
    {"Task": "Feature Extraction", "Start": 8, "Duration": 3, "Phase": "Phase 3"},
    {"Task": "Data Merging & Splitting", "Start": 10, "Duration": 2, "Phase": "Phase 3"},
    
    # Phase 4: Training
    {"Task": "Model Architecture Design", "Start": 12, "Duration": 2, "Phase": "Phase 4"},
    {"Task": "Training & Validation", "Start": 13, "Duration": 4, "Phase": "Phase 4"},
    
    # Phase 5: App Dev & Testing
    {"Task": "Application UI Development", "Start": 17, "Duration": 4, "Phase": "Phase 5"},
    {"Task": "Final Testing & Deployment", "Start": 19, "Duration": 3, "Phase": "Phase 5"},
]

df = pd.DataFrame(data)

# Color mapping for phases
colors = {
    "Phase 1": "#E64A19",
    "Phase 2": "#FBC02D",
    "Phase 3": "#7CB342",
    "Phase 4": "#0288D1",
    "Phase 5": "#5E35B1"
}

fig, ax = plt.subplots(figsize=(14, 8))

# Plot bars
for i, task in enumerate(reversed(df['Task'].unique())):
    task_data = df[df['Task'] == task]
    for _, row in task_data.iterrows():
        ax.barh(i, row['Duration'], left=row['Start'], height=0.5, 
                color=colors[row['Phase']], edgecolor='black', alpha=0.9)

# Formatting Y-axis
ax.set_yticks(range(len(df['Task'].unique())))
ax.set_yticklabels(reversed(df['Task'].unique()), fontsize=12)

# Formatting X-axis mechanism
# Calculate phase boundaries to place X-axis labels roughly where each phase dominates
# Since phases are roughly sequential, we can just find the min start and max end for each.
phase_labels = []
phase_ticks = []

sorted_phases = sorted(list(colors.keys()))
for phase in sorted_phases:
    phase_data = df[df['Phase'] == phase]
    if not phase_data.empty:
        start = phase_data['Start'].min()
        end = (phase_data['Start'] + phase_data['Duration']).max()
        midpoint = start + (end - start) / 2
        phase_labels.append(phase)
        phase_ticks.append(midpoint)

ax.set_xticks(phase_ticks)
ax.set_xticklabels(phase_labels, fontsize=12, fontweight='bold')
ax.set_xlabel("Project Phases", fontsize=14, labelpad=10)

# Add grid lines for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.3)

# Title
plt.title('SignVision S8 V2 - Progression Timeline', fontsize=18, pad=20)

plt.tight_layout()

# Save and show
output_file = 'project_timeline.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Gantt chart saved to {output_file}")
