from pstats import Stats
from pstats import SortKey

p = Stats("graphs_profiling")
p.strip_dirs()
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(20, "create_graphs_constantin.py")
# p.print_callers(20, "create_graphs_constantin.py")
p.print_callees(20, "create_graphs_constantin.py")
