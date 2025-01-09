import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_actions(objects, actions):
    """
    Plots a bar chart of actions performed for each object.

    Parameters:
        objects (list of str): List of object names.
        actions (list of tuples): List of actions for each object. Each action is a tuple or string.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define some colors for actions
    action_colors = {
        'pref': '#c9b35b',
        'demo': '#9a4f69',
        'act': '#b39c85',
        'human': '#69755c',
    }

    # Define action order
    action_order = ['pref', 'demo', 'act', 'human']

    # Create bars for each object and its corresponding actions
    for idx, (obj, action) in enumerate(zip(objects, actions)):
        start = idx  # Bar starts at the index
        width = 1  # Width of the bar spans one unit

        if isinstance(action, tuple):
            # Split the bar if multiple actions are present
            half_width = width / 2
            ax.barh(y=action_order.index(action[0]), width=half_width, left=start, color=action_colors[action[0]], edgecolor='black', align='center')
            ax.barh(y=action_order.index(action[1]), width=half_width, left=start + half_width, color=action_colors[action[1]], edgecolor='black', align='center')
        else:
            # Single action bar
            ax.barh(y=action_order.index(action), width=width, left=start, color=action_colors[action], edgecolor='black', align='center')

    # Set the x-axis to show object names
    ax.set_xticks(range(len(objects)))
    ax.set_xticklabels(objects)

    # Set the y-axis to show action names
    ax.set_yticks(range(len(action_order)))
    ax.set_yticklabels(action_order)

    # Set the y-axis label and x-axis label
    ax.set_ylabel('Action Name')
    ax.set_xlabel('Object Name')
    ax.set_title('Actions Performed on Objects')

    # Add legend
    patches = [mpatches.Patch(color=color, label=action) for action, color in action_colors.items()]
    ax.legend(handles=patches, title="Actions")

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
objects = ['red key', 'blue key', 'red key', 'red key', 'yellow ball']
actions = [('pref', 'demo'), 'human', 'act', 'act']
plot_actions(objects, actions)

