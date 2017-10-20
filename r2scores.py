import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 5

means_men = (0.26, 0.55450033722, 0.0390325670498, 0.00011973, 0.00634578544061)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_men, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config)

ax.set_xlabel('Algorithm')
ax.set_ylabel('Scores')
ax.set_title('R^2 Regression Scores by Algorithm')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Linear Regression', 'Decision Tree', 'Random Forest', 'Deep Learning', 'Dummy'))
ax.legend()

fig.tight_layout()
plt.show()
