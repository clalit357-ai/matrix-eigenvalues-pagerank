"""
graph.py : graph construction, adjacency matrix, sample graphs, visualization.

CONVENTION (used everywhere in this project):
  - Adjacency matrix A is row-based:  A[i, j] = 1  if page i links TO page j.
  - Transition matrix M (in pagerank.py) is COLUMN-stochastic:
        M[i, j] = probability of moving from page j to page i
    so every column sums to 1, and PageRank is a column vector with  PR = G @ PR.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------- construction ----------
def build_graph(edges, nodes=None):
    """Directed graph from a list of (src, dst) edges.
    Pass `nodes` to include pages that have no links at all (isolated)."""
    G = nx.DiGraph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def adjacency_matrix(G, nodelist=None):
    """Return (A, nodelist). A[i, j] = 1 if nodelist[i] -> nodelist[j]."""
    if nodelist is None:
        nodelist = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=None, dtype=float)
    return A, list(nodelist)


# ---------- sample graphs ----------
def example_graph():
    """4-node graph from the roadmap: A->B, A->C, B->C, C->A, C->D, D->A"""
    return build_graph([("A", "B"), ("A", "C"), ("B", "C"),
                        ("C", "A"), ("C", "D"), ("D", "A")])


def dangling_graph():
    """A->B, B->C, C has no outgoing links (dangling node)."""
    return build_graph([("A", "B"), ("B", "C")])


def spider_trap_graph():
    """C and D only link to each other -> they trap rank."""
    return build_graph([("A", "B"), ("A", "C"), ("B", "C"),
                        ("C", "D"), ("D", "C")])


def chain_graph(n=5):
    return build_graph([(i, i + 1) for i in range(n - 1)])


def cycle_graph(n=5):
    return build_graph([(i, (i + 1) % n) for i in range(n)])


def star_graph(n=5):
    """Everyone links to node 0 (in-star)."""
    return build_graph([(i, 0) for i in range(1, n)] + [(0, 1)])


def hub_graph(n=6):
    """Node 0 links to everyone, everyone links back to 0."""
    edges = [(0, i) for i in range(1, n)] + [(i, 0) for i in range(1, n)]
    return build_graph(edges)


def six_node_graph():
    return build_graph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
                        ("C", "E"), ("D", "A"), ("E", "F"), ("F", "C"),
                        ("F", "A")])


# ---------- visualization ----------
def draw_graph(G, pagerank=None, nodelist=None, title="Web graph",
               save_path=None, seed=42):
    """Draw directed graph; node size and color scale with PageRank if given."""
    if nodelist is None:
        nodelist = sorted(G.nodes())
    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 5))

    if pagerank is not None:
        pr = np.asarray(pagerank, dtype=float)
        sizes = 600 + 5000 * pr
        colors = pr
    else:
        sizes, colors = 900, "#9ecae1"

    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=sizes,
                                   node_color=colors, cmap=plt.cm.YlOrRd,
                                   edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                           node_size=sizes, connectionstyle="arc3,rad=0.1",
                           ax=ax)
    labels = {n: f"{n}\n{pagerank[i]:.3f}" if pagerank is not None else str(n)
              for i, n in enumerate(nodelist)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    if pagerank is not None:
        plt.colorbar(nodes, ax=ax, label="PageRank")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_pagerank_bars(nodelist, pagerank, title="PageRank scores", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(n) for n in nodelist], pagerank, color="#fd8d3c", edgecolor="black")
    ax.set_xlabel("Page")
    ax.set_ylabel("PageRank")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_convergence(histories, labels=None, title="Power iteration convergence",
                     save_path=None):
    """histories: list of error lists (one per run)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.semilogy(range(1, len(h) + 1), h,
                    label=labels[i] if labels else None, marker="o", ms=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|PR^{(k+1)} - PR^{(k)}\|_1$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if labels:
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
