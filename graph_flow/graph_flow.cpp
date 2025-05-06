#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lp_lib.h"

namespace py = pybind11;


#include <set>
#include <vector>
#include <limits>
#include <iostream>

const double INF = std::numeric_limits<double>::max();

struct Edge
{
    int u, v;
    double flow = 0, max_flow;
    double cost;
    double add_cost = 0;
    double time = -1;
    bool free = false;
    double weight = 0;
    std::vector<int> corr_components;
    int def_corr = -1;

    Edge(int u, int v, double max_flow, double cost);
    double get_cost() const;
    void add_flow(double f);
};

Edge::Edge(int u, int v, double max_flow, double cost) : u(u),
                                                         v(v),
                                                         max_flow(max_flow),
                                                         cost(cost)
{
}

inline double Edge::get_cost() const
{
    return cost + add_cost;
}

inline void Edge::add_flow(double f)
{
    flow += f;
}

class Graph
{
public:
    int n;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> out_edges;
    std::vector<std::vector<int>> in_edges;

    Graph(int n, double eps, std::vector<std::tuple<int, int, double, double>> edges) : n(n),
                                                                                        out_edges(n),
                                                                                        in_edges(n),
                                                                                        eps(eps)
    {
        for (auto &e : edges)
        {
            const auto [u, v, max_flow, cost] = e;
            this->add_edge(u, v, max_flow, cost);
        }
    }
    int add_edge(int u, int v, double max_flow, double cost);
    void add_cost(int edge_id, double cost);
    void reset_cost(int edge_id);
    void reset_costs();
    std::vector<int> get_path(int f_id);
    void set_flow(const std::vector<std::tuple<int, int, double>> &flow);
    py::tuple calculate_flow();
    py::tuple calculate_time();
    void reset_flow();

private:
    struct Corr
    {
        int u, v;
        float flow;
        int component;
        Corr(int u, int v, float f) : u(u), v(v), flow(f), component(-1)
        {
        }
    };

    struct DSUEdges
    {
        int corr;
        std::vector<int> parents;
        std::vector<int> rank;
        std::vector<std::vector<int>> edge_list;
        std::vector<std::vector<int>> node_list;
        std::vector<Edge> &edges;
        std::vector<double> potential;

        DSUEdges(int n, int c, std::vector<Edge> &e);
        int get(int u);
        bool merge(Edge &e, std::vector<int> &edge_queue);
    };

    class SumTree
    {
    public:
        SumTree(const std::vector<std::vector<int>> &neighb,
                std::vector<Edge> &edges);
        int lca(int u, int v);
        void add_flow(int u, int v, double w);
        double get_edge(int k);
        std::vector<int> parent_edges;

    private:
        double get_all(int u);
        void dfs(int u, const std::vector<std::vector<int>> &neighb, std::vector<int> &order);
        int lca_tree(int u, int v, int k, int l, int r);
        void add_value(int u, double w);
        int z;

        std::vector<Edge> &edges;
        std::vector<int> directions;

        std::vector<int> h, tree, first, last;
        std::vector<std::vector<int>> p;
        std::vector<double> f;
    };

    float eps;
    std::vector<Corr> d_flow;
    std::vector<std::vector<int>> d_corrs;
    std::vector<std::vector<int>> corr_edges;
    std::vector<int> corrs_start;
    std::vector<DSUEdges> dsu;
    std::vector<SumTree> corr_trees;

    int corr_components;

    void set_dsu();
    void set_dsu_free_edges(std::vector<int> &edges_queue,
                            std::vector<std::vector<std::vector<int>>> &corr_edges);
    void process_queue(std::vector<int> &edges_queue, int &queue_pos,
                       std::vector<std::vector<std::vector<int>>> &corr_edges);
    bool flow_linear();
    std::vector<int> get_path_uv(int u, int v, int corr);
    std::vector<int> get_path_up_uv(int u, int v, int corr);
};

int Graph::add_edge(int u, int v, double max_flow, double cost)
{
    u--;
    v--;
    out_edges[u].push_back(edges.size());
    in_edges[v].push_back(edges.size());
    edges.emplace_back(u, v, max_flow, cost);
    return edges.size() - 1;
}

void Graph::add_cost(int id, double cost)
{
    edges[id].add_cost += cost;
}

void Graph::reset_cost(int id)
{
    edges[id].add_cost = 0;
}

void Graph::reset_costs()
{
    for (auto &e : edges)
    {
        e.add_cost = 0;
    }
}

void Graph::reset_flow()
{
    for (auto &e : edges)
    {
        e.flow = 0;
        e.free = false;
        e.weight = 0;
        e.corr_components.clear();
        e.time = -1;
        e.def_corr = -1;
    }
    corr_trees.clear();
    dsu.clear();
}

void Graph::set_flow(const std::vector<std::tuple<int, int, double>> &flow)
{
    reset_flow();
    d_flow.clear();
    d_corrs.clear();
    corrs_start.clear();
    corr_edges.clear();

    std::vector<std::vector<int>> conn(n);
    for (auto &s : flow)
    {
        auto [u, v, f] = s;
        conn[u - 1].push_back(d_flow.size());
        d_flow.push_back(Corr(u - 1, v - 1, f));
    }

    corr_components = 0;
    for (int i = 0; i < conn.size(); i++)
    {
        auto conns = conn[i];
        if (!conns.empty())
        {
            for (auto i : conns)
            {
                d_flow[i].component = corr_components;
            }
            corr_components += 1;
            d_corrs.push_back(std::move(conns));
            corrs_start.push_back(i);
        }
    }
    corr_edges = std::vector<std::vector<int>>(corr_components);
}

bool Graph::flow_linear()
{
    int edge_cols = d_corrs.size();
    int n_cols = edge_cols * edges.size();
    lprec *lp = make_lp(0, n_cols);
    if (lp == NULL)
    {
        return false;
    }
    std::vector<std::string> labels(d_flow.size());
    int cnt = 0;
    for (int k = 0; k < edges.size(); k++)
    {
        for (auto u : corrs_start)
        {
            cnt++;
            labels.push_back("(" + std::to_string(u) + "|" + std::to_string(edges[k].u + 1) + "->" + std::to_string(edges[k].v + 1) + ")");
            set_col_name(lp, cnt, labels.back().data());
        }
    }
    auto col_no = (int *)calloc(edge_cols, sizeof(int));
    auto row = (REAL *)calloc(edge_cols, sizeof(REAL));
    for (int i = 0; i < edge_cols; i++)
    {
        row[i] = 1;
    }
    if ((col_no == NULL) || (row == NULL))
    {
        if (col_no != NULL)
        {
            free(col_no);
        }
        if (row != NULL)
        {
            free(row);
        }
        delete_lp(lp);
        return false;
    }
    cnt = 0;
    for (int k = 0; k < edges.size(); k++)
    {
        for (int i = 0; i < edge_cols; i++)
        {
            cnt++;
            // if (edges[k].blocked)
            //{
            //     set_bounds(lp, cnt, 0, 0);
            // }
        }
    }
    set_add_rowmode(lp, TRUE);
    for (int k = 0; k < edges.size(); k++)
    {
        for (int i = 0; i < edge_cols; i++)
        {
            col_no[i] = k * edge_cols + i + 1;
        }
        if (!add_constraintex(lp, edge_cols, row, col_no, LE, edges[k].max_flow))
        {
            free(col_no);
            free(row);
            delete_lp(lp);
            return false;
        }
    }
    free(col_no);
    free(row);
    col_no = (int *)calloc(edges.size(), sizeof(int));
    row = (REAL *)calloc(edges.size(), sizeof(REAL));
    if ((col_no == NULL) || (row == NULL))
    {
        if (col_no != NULL)
        {
            free(col_no);
        }
        if (row != NULL)
        {
            free(row);
        }
        delete_lp(lp);
        return false;
    }
    cnt = 0;
    for (int t = 0; t < edges.size(); t++)
    {
        col_no[t] = 0;
        row[t] = 0;
    }
    for (int i = 0; i < d_corrs.size(); i++)
    {
        std::vector<int> corr_finish(n, -1);
        for (auto c : d_corrs[i])
        {
            corr_finish[d_flow[c].v] = c;
        }
        for (int k = 0; k < n; k++)
        {
            for (int t = 0; t < out_edges[k].size(); t++)
            {
                col_no[t] = out_edges[k][t] * edge_cols + cnt + 1;
                row[t] = 1.;
            }
            for (int t = 0; t < in_edges[k].size(); t++)
            {
                col_no[t + out_edges[k].size()] = in_edges[k][t] * edge_cols + cnt + 1;
                row[t + out_edges[k].size()] = -1.;
            }
            double sum = 0;
            if (k == corrs_start[i])
            {
                for (auto c : d_corrs[i])
                {
                    sum += d_flow[c].flow;
                }
            }
            else
            {
                if (corr_finish[k] != -1)
                {
                    sum = -d_flow[corr_finish[k]].flow;
                }
            }
            if (!add_constraintex(lp, in_edges[k].size() + out_edges[k].size(), row, col_no, EQ, sum))
            {
                free(col_no);
                free(row);
                delete_lp(lp);
                return false;
            }
        }
        cnt += 1;
    }
    set_add_rowmode(lp, FALSE);
    free(col_no);
    free(row);
    col_no = (int *)calloc(n_cols, sizeof(int));
    row = (REAL *)calloc(n_cols, sizeof(REAL));
    cnt = 0;
    for (int k = 0; k < edges.size(); k++)
    {
        for (int i = 0; i < edge_cols; i++)
        {
            row[cnt] = edges[k].get_cost();
            col_no[cnt] = cnt + 1;
            cnt++;
        }
    }
    if (!set_obj_fnex(lp, n_cols, row, col_no))
    {
        free(col_no);
        free(row);
        delete_lp(lp);
        return false;
    }
    set_minim(lp);
    set_verbose(lp, IMPORTANT);

    // now let lpsolve calculate a solution
    // std::cout << "solving..." << std::endl;
    // write_LP(lp, stdout);
    if (solve(lp) != OPTIMAL)
    {
        free(col_no);
        free(row);
        delete_lp(lp);
        return false;
    }
    // print_solution(lp, 1);
    // variable values
    get_variables(lp, row);
    cnt = 0;
    for (int k = 0; k < edges.size(); k++)
    {
        edges[k].flow = 0;
    }
    std::vector<double> balance(n);
    for (int k = 0; k < edges.size(); k++)
    {
        for (int i = 0; i < corr_components; i++)
        {
            if (row[cnt] > eps)
            {
                if (i == 0)
                {
                    balance[edges[k].u] += row[cnt];
                    balance[edges[k].v] -= row[cnt];
                }
                edges[k].flow += row[cnt];
                edges[k].corr_components.push_back(i);
                corr_edges[i].push_back(k);
            }
            cnt++;
        }
        if (edges[k].flow + eps < edges[k].max_flow && edges[k].flow > eps)
        {
            edges[k].free = true;
        }
    }

    free(col_no);
    free(row);
    delete_lp(lp);

    return true;
}

Graph::DSUEdges::DSUEdges(int n, int c, std::vector<Edge> &e) : parents(n), rank(n),
                                                                corr(c), edge_list(n),
                                                                edges(e), potential(n),
                                                                node_list(n)
{
    for (int i = 0; i < n; i++)
    {
        parents[i] = i;
        node_list[i].push_back(i);
    }
}

int Graph::DSUEdges::get(int u)
{
    if (parents[u] != u)
        parents[u] = get(parents[u]);
    return parents[u];
}

bool Graph::DSUEdges::merge(Edge &e, std::vector<int> &edge_queue)
{
    int u = get(e.u);
    int v = get(e.v);
    if (u == v)
        return false;

    std::vector<int> d_vertices;
    std::vector<int> h_vertices;

    if (node_list[v].size() > node_list[u].size())
    {
        double p_u = potential[e.v] - e.time - potential[e.u];
        for (auto k : node_list[u])
        {
            potential[k] += p_u;
        }
        h_vertices = std::move(node_list[u]);
        d_vertices = std::move(node_list[v]);
    }
    else
    {
        double p_v = potential[e.u] + e.time - potential[e.v];
        for (auto k : node_list[v])
        {
            potential[k] += p_v;
        }
        h_vertices = std::move(node_list[v]);
        d_vertices = std::move(node_list[u]);
    }
    for (auto x : h_vertices)
    {
        d_vertices.push_back(x);
    }
    h_vertices.clear();

    // ====================================================
    std::vector<int> h_edges = std::move(edge_list[u]);
    std::vector<int> d_edges = std::move(edge_list[v]);
    if (h_edges.size() < d_edges.size())
    {
        std::swap(h_edges, d_edges);
    }
    for (auto e : d_edges)
    {

        int u_h = get(edges[e].u), v_h = get(edges[e].v);
        if ((u == u_h && v == v_h) || (u == v_h && v == u_h))
        {
            edges[e].time = potential[edges[e].v] - potential[edges[e].u];
            if (edges[e].def_corr == -1)
            {
                edges[e].def_corr = corr;
                edge_queue.push_back(e);
            }
        }
        else
            h_edges.push_back(e);
    }
    d_edges.clear();

    if (rank[u] == rank[v])
        rank[u] += 1;
    if (rank[u] > rank[v])
    {
        parents[v] = u;
        edge_list[u] = std::move(h_edges);
        node_list[u] = std::move(d_vertices);
    }
    else
    {
        parents[u] = v;
        edge_list[v] = std::move(h_edges);
        node_list[v] = std::move(d_vertices);
    }

    return true;
}

void Graph::set_dsu()
{
    for (int i = 0; i < corr_components; i++)
    {
        dsu.push_back(DSUEdges(n, i, edges));
    }
    for (int i = 0; i < edges.size(); i++)
    {
        for (auto e : edges[i].corr_components)
        {
            if (!edges[i].free)
            {
                dsu[e].edge_list[edges[i].u].push_back(i);
                dsu[e].edge_list[edges[i].v].push_back(i);
            }
        }
    }
}

void Graph::set_dsu_free_edges(std::vector<int> &edges_queue,
                               std::vector<std::vector<std::vector<int>>> &corr_edges)
{
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].free)
        {
            edges[i].time = edges[i].get_cost();
            for (auto e : edges[i].corr_components)
            {
                if (dsu[e].merge(edges[i], edges_queue))
                {
                    corr_edges[e][edges[i].u].push_back(i);
                    corr_edges[e][edges[i].v].push_back(i);
                }
            }
        }
    }
}

void Graph::process_queue(std::vector<int> &edges_queue, int &queue_pos,
                          std::vector<std::vector<std::vector<int>>> &corr_edges)
{
    while (queue_pos < edges_queue.size())
    {
        int e = edges_queue[queue_pos];
        auto &edge = edges[e];
        for (auto &corr : edge.corr_components)
        {
            if (corr != edge.def_corr && dsu[corr].merge(edge, edges_queue))
            {
                corr_edges[corr][edge.u].push_back(e);
                corr_edges[corr][edge.v].push_back(e);
            }
        }
        queue_pos++;
    }
    for (int i = 0; i < edges.size(); i++)
    {
        if (!edges[i].free && edges[i].flow > eps && edges[i].def_corr == -1)
        {
            std::cout << "not processed: " << i << " " << edges[i].u << "->" << edges[i].v << std::endl;
        }
    }
}

py::tuple Graph::calculate_flow()
{
    if (flow_linear())
    {
        std::vector<py::tuple> free_edges(edges.size());
        for (int i = 0; i < edges.size(); i++)
        {
            free_edges[i] = py::make_tuple(edges[i].free, edges[i].flow < eps, edges[i].flow);
        }
        return py::make_tuple(true, free_edges, corr_edges);
    }
    else
    {
        return py::make_tuple(false, py::none(), py::none());
    }
}

py::tuple Graph::calculate_time()
{
    set_dsu();
    std::vector<bool> used_edges(edges.size());
    auto corr_edges = std::vector<std::vector<std::vector<int>>>(corr_components,
                                                                 std::vector<std::vector<int>>(n));

    int pos = 0;
    std::vector<int> edges_order;

    set_dsu_free_edges(edges_order, corr_edges);
    process_queue(edges_order, pos, corr_edges);

    for (int j = 0; j < corr_components; j++)
    {
        int k = -1;
        for (int i = 0; i < n; i++)
        {
            int t = dsu[j].get(i);
            if (t != i && t != k)
            {
                if (k == -1)
                    k = t;
                else
                {
                    return py::make_tuple(false, std::vector<int>(), std::vector<double>(), std::vector<double>(), 0);
                }
            }
        }
    }

    for (auto &c_e : corr_edges)
    {
        corr_trees.push_back(SumTree(c_e, edges));
    }

    for (auto &cor : d_flow)
    {
        corr_trees[cor.component].add_flow(cor.u, cor.v, cor.flow);
    }
    for (int i = edges_order.size() - 1; i >= 0; i--)
    {
        int ord = edges_order[i];
        for (auto &tree : corr_trees)
        {
            edges[ord].weight += tree.get_edge(ord);
        }
        corr_trees[edges[ord].def_corr].add_flow(edges[ord].u, edges[ord].v, edges[ord].weight);
    }
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].free)
        {
            for (auto &tree : corr_trees)
            {
                edges[i].weight += tree.get_edge(i);
            }
        }
    }
    std::vector<py::tuple> braess;
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].weight < 0 && edges[i].free)
            braess.push_back(py::make_tuple(i, edges[i].weight));
    }
    std::vector<double> costs;
    double sum;
    for (auto &f : d_flow)
    {
        costs.push_back(dsu[f.component].potential[f.v] - dsu[f.component].potential[f.u]);
        sum += f.flow * costs[costs.size() - 1];
    }
    std::vector<int> e_add;
    for (auto &e : edges)
    {
        e_add.push_back(e.add_cost);
    }
    return py::make_tuple(true, braess, costs, e_add, sum);
}

std::vector<int> Graph::get_path(int f_id)
{
    return get_path_uv(d_flow[f_id].u, d_flow[f_id].v, d_flow[f_id].component);
}

std::vector<int> Graph::get_path_uv(int u, int v, int corr)
{
    int w = corr_trees[corr].lca(u, v);
    auto r_u = get_path_up_uv(u, w, corr);
    auto r_v = get_path_up_uv(v, w, corr);
    r_u.insert(r_u.end(), r_v.rbegin(), r_v.rend());
    return r_u;
}

std::vector<int> Graph::get_path_up_uv(int u, int v, int corr)
{
    std::vector<int> res_u;
    while (u != v)
    {
        int e = corr_trees[corr].parent_edges[u];
        int h_u = edges[e].v;
        if (h_u == u)
        {
            h_u = edges[e].u;
        }
        if (edges[e].free)
        {
            res_u.push_back(e);
        }
        else
        {
            auto r_edges = get_path_uv(u, h_u, edges[e].def_corr);
            res_u.insert(res_u.end(), r_edges.begin(), r_edges.end());
        }
        u = h_u;
    }
    return res_u;
}

Graph::SumTree::SumTree(const std::vector<std::vector<int>> &neighb,
                        std::vector<Edge> &edges) : parent_edges(neighb.size()),
                                                    directions(edges.size()),
                                                    h(neighb.size(), -1),
                                                    first(neighb.size(), -1),
                                                    last(neighb.size(), -1),
                                                    f(neighb.size() * 2 - 1),
                                                    edges(edges)
{
    std::vector<int> sorted;
    sorted.reserve(neighb.size() * 2 - 1);
    for (int i = 0; i < neighb.size(); i++)
    {
        if (neighb[i].size() > 0)
        {
            dfs(i, neighb, sorted);
            break;
        }
    }

    z = 1;
    while (z < sorted.size())
    {
        z <<= 1;
    }
    tree = std::vector<int>(z * 2 - 1, -1);
    for (int i = 0; i < sorted.size(); i++)
        tree[z - 1 + i] = sorted[i];

    for (int i = z - 2; i >= 0; i--)
    {
        if (tree[2 * i + 2] == -1 || h[tree[2 * i + 1]] < h[tree[2 * i + 2]])
        {
            tree[i] = tree[2 * i + 1];
        }
        else
        {
            tree[i] = tree[2 * i + 2];
        }
    }
}

void Graph::SumTree::dfs(int u, const std::vector<std::vector<int>> &neighb, std::vector<int> &order)
{
    first[u] = order.size();
    order.push_back(u);
    for (auto &e : neighb[u])
    {
        int k = 1, v = edges[e].u;
        if (v == u)
        {
            k = -1;
            v = edges[e].v;
        }
        if (h[v] == -1)
        {
            parent_edges[v] = e;
            directions[e] = k;
            h[v] = h[u] + 1;
            dfs(v, neighb, order);
            order.push_back(u);
        }
    }
    last[u] = order.size() - 1;
}

int Graph::SumTree::lca(int u, int v)
{
    int p = first[u];
    int q = first[v];
    if (p > q)
    {
        std::swap(p, q);
    }
    return lca_tree(p, q + 1, 0, 0, z);
}

int Graph::SumTree::lca_tree(int u, int v, int k, int l, int r)
{
    if (tree[k] == -1 || (u <= l && v >= r))
    {
        return tree[k];
    }
    int m = (l + r) / 2;
    if (v <= m)
    {
        return lca_tree(u, v, k * 2 + 1, l, m);
    }
    else if (u >= m)
    {
        return lca_tree(u, v, k * 2 + 2, m, r);
    }
    else
    {
        int a = lca_tree(u, v, k * 2 + 1, l, m);
        int b = lca_tree(u, v, k * 2 + 2, m, r);
        if (b == -1 || h[a] < h[b])
        {
            return a;
        }
        else
        {
            return b;
        }
    }
}

void Graph::SumTree::add_flow(int u, int v, double w)
{
    add_value(first[u], w);
    add_value(first[v], -w);
}

void Graph::SumTree::add_value(int u, double w)
{
    for (; u < f.size(); u = u | (u + 1))
        f[u] += w;
}

double Graph::SumTree::get_all(int u)
{
    double ret = 0;
    for (; u >= 0; u = (u & (u + 1)) - 1)
        ret += f[u];
    return ret;
}

double Graph::SumTree::get_edge(int k)
{
    auto &e = edges[k];
    int d = directions[k];
    if (d != 0)
    {
        int u = e.u;
        if (d == -1)
        {
            u = e.v;
        }
        return (get_all(last[u]) - get_all(first[u] - 1)) * d;
    }
    else
    {
        return 0;
    }
}

PYBIND11_MODULE(graph_flow, m)
{
    py::class_<Graph>(m, "Graph")
        .def(py::init<int, double, std::vector<std::tuple<int, int, double, double>>>(),
             py::arg("n"), py::arg("eps") = 1e-9,
             py::arg("edges") = std::vector<std::tuple<int, int, double, double>>())
        .def("add_edge", &Graph::add_edge)
        .def("add_cost", &Graph::add_cost)
        .def("reset_cost", &Graph::reset_cost)
        .def("reset_costs", &Graph::reset_costs)
        .def("reset_flow", &Graph::reset_flow)
        .def("set_flow", &Graph::set_flow)
        .def("calculate_flow", &Graph::calculate_flow)
        .def("calculate_time", &Graph::calculate_time)
        .def("get_path", &Graph::get_path);
}
