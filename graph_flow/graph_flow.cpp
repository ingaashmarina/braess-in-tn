#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lp_lib.h"

namespace py = pybind11;

#include <set>
#include <vector>
#include <limits>
#include <iostream>

const double INF = std::numeric_limits<double>::max();

class Edge
{
public:
    int u, v;
    double flow = 0, max_flow;
    double cost;
    bool free = false;
    bool blocked = false;
    double weight = 0;
    std::vector<int> corr_components;

    Edge(int u, int v, double max_flow, double cost);
    double get_residual_capacity() const;

    void add_flow(double f);
};

Edge::Edge(int u, int v, double max_flow, double cost) : u(u),
                                                         v(v),
                                                         max_flow(max_flow),
                                                         cost(cost)
{
}

inline double Edge::get_residual_capacity() const
{
    return max_flow - flow;
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
    void block_edge(int id);
    void add_cost(int id, double flow);
    void reuse_edge(int id);
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
        const std::vector<Edge> &edges;
        DSUEdges(int n, int c, const std::vector<Edge> &e) : parents(n), rank(n),
                                                             corr(c), edge_list(n),
                                                             edges(e)
        {
            for (int i = 0; i < n; i++)
            {
                parents[i] = i;
            }
        }
        int get(int u)
        {
            if (parents[u] != u)
                parents[u] = get(parents[u]);
            return parents[u];
        }
        bool merge(int u, int v, std::vector<bool> &used_edges,
                   std::deque<std::pair<int, int>> &edge_queue)
        {
            u = get(u);
            v = get(v);
            if (u == v)
                return false;

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
                    if (!used_edges[e])
                    {
                        used_edges[e] = true;
                        edge_queue.push_back({corr, e});
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
            }
            else
            {
                parents[u] = v;
                edge_list[v] = std::move(h_edges);
            }
            return true;
        }
    };

    class SumTree
    {
    public:
        SumTree(const std::vector<std::vector<int>> &neighb);
        int lca(int u, int v);
        void add_flow(int u, int v, double w);
        double get(int u);

    private:
        double get_all(int u);
        void dfs(int u, const std::vector<std::vector<int>> &neighb, std::vector<int> &order);
        int lca_tree(int u, int v, int k, int l, int r);
        void add_value(int u, double w);
        int z;
        std::vector<int> parents, h, tree, first, last;
        std::vector<std::vector<int>> p;
        std::vector<double> f, cost;
    };

    float eps;
    std::vector<Corr> d_flow;
    std::vector<std::vector<int>> d_corrs;
    std::vector<int> corrs_start;
    std::vector<DSUEdges> dsu;

    std::vector<std::vector<std::vector<std::pair<int, int>>>> corr_edges;
    int corr_components;

    void set_dsu();
    void set_dsu_free_edges(std::vector<bool> &used_edges, std::deque<std::pair<int, int>> &edges_queue);
    void process_queue(std::vector<bool> &used_edges, std::deque<std::pair<int, int>> &edges_queue);
    bool flow_linear();
    bool find_cost_corr(int u, int v, int corr, std::vector<bool> &vis_vert,
                        std::vector<std::pair<int, std::pair<int, int>>> &edges_stack)
    {
        vis_vert[u] = true;
        if (u == v)
        {
            return true;
        }
        else
        {
            for (auto &[e_corr, e] : corr_edges[corr][u])
            {
                int w, k;
                if (edges[e].u == u)
                {
                    w = edges[e].v;
                    k = 1;
                }
                else
                {
                    w = edges[e].u;
                    k = -1;
                }
                if (!vis_vert[w])
                {
                    edges_stack.push_back({k, {e_corr, e}});
                    if (find_cost_corr(w, v, corr, vis_vert, edges_stack))
                    {
                        return true;
                    }
                    edges_stack.pop_back();
                }
            }
        }
        return false;
    }
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

void Graph::add_cost(int id, double flow)
{
    edges[id].cost += flow;
}

void Graph::block_edge(int id)
{
    edges[id].blocked = true;
}

void Graph::reuse_edge(int id)
{
    edges[id].blocked = false;
}

void Graph::reset_flow()
{
    for (auto &e : edges)
    {
        e.flow = 0;
        e.free = false;
        e.weight = 0;
        e.corr_components.clear();
    }
}

void Graph::set_flow(const std::vector<std::tuple<int, int, double>> &flow)
{
    reset_flow();
    d_flow.clear();
    d_corrs.clear();
    corrs_start.clear();

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
            if (edges[k].blocked)
            {
                set_bounds(lp, cnt, 0, 0);
            }
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
            row[cnt] = edges[k].cost;
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
        std::cout << "failure" << std::endl;
        return false;
    }
    // print_solution(lp, 1);
    std::cout << "success" << std::endl;
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
            }
            cnt++;
        }
        if (edges[k].flow + eps < edges[k].max_flow)
        {
            edges[k].free = true;
        }
    }
    free(col_no);
    free(row);
    delete_lp(lp);

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

void Graph::set_dsu_free_edges(std::vector<bool> &used_edges, std::deque<std::pair<int, int>> &edges_queue)
{
    corr_edges = std::vector<std::vector<std::vector<std::pair<int, int>>>>(corr_components,
                                                                            std::vector<std::vector<std::pair<int, int>>>(n));
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].free)
        {
            used_edges[i] = true;
            for (auto e : edges[i].corr_components)
            {
                if (dsu[e].merge(edges[i].u, edges[i].v, used_edges, edges_queue))
                {
                    corr_edges[e][edges[i].u].push_back({e, i});
                    corr_edges[e][edges[i].v].push_back({e, i});
                }
            }
        }
    }
}

void Graph::process_queue(std::vector<bool> &used_edges, std::deque<std::pair<int, int>> &edges_queue)
{
    while (!edges_queue.empty())
    {
        auto [p_corr, e] = edges_queue.front();
        auto &edge = edges[e];
        for (auto &corr : edge.corr_components)
        {
            if (corr != p_corr && dsu[corr].merge(edge.u, edge.v, used_edges, edges_queue))
            {
                corr_edges[corr][edge.u].push_back({p_corr, e});
                corr_edges[corr][edge.v].push_back({p_corr, e});
            }
        }
        edges_queue.pop_front();
    }
    for (int i = 0; i < used_edges.size(); i++)
    {
        if (!used_edges[i])
        {
            std::cout << "not processed: " << i << " " << edges[i].u << "->" << edges[i].v << std::endl;
        }
    }
}

py::tuple Graph::calculate_flow()
{

    std::cout << "calculating..." << std::endl;
    if (flow_linear())
    {
        std::vector<py::tuple> free_edges(edges.size());
        for (int i = 0; i < edges.size(); i++)
        {
            free_edges[i] = py::make_tuple(edges[i].free, edges[i].flow);
        }
        return py::make_tuple(true, free_edges);
    }
    else
    {
        return py::make_tuple(false, py::none());
    }
}

py::tuple Graph::calculate_time()
{
    set_dsu();
    std::vector<bool> used_edges(edges.size());
    std::deque<std::pair<int, int>> edges_queue;
    set_dsu_free_edges(used_edges, edges_queue);
    process_queue(used_edges, edges_queue);

    std::vector<std::vector<int>> c_edges(corr_components);
    for (int i = 0; i < corr_components; i++)
    {
        std::set<int> w;
        for (auto &l : corr_edges[i])
        {
            for (auto &p : l)
            {
                for (auto &z : l)
                {
                    w.insert(z.second);
                }
            }
        }
        c_edges[i] = std::vector<int>(w.begin(), w.end());
    }
    //

    std::vector<double> costs;
    double sum;
    std::cout << "paths" << std::endl;
    for (auto &cor : d_flow)
    {
        std::vector<std::vector<bool>> used(corr_components, std::vector<bool>(n));
        std::deque<std::tuple<int, int, int>> queue{{cor.u, cor.v, cor.component}};
        double f = 0;
        while (!queue.empty())
        {
            auto [u, v, c] = queue.front();
            // std::cout << "edges " << u << " " << v << " " << c << std::endl;
            std::vector<std::pair<int, std::pair<int, int>>> edges_stack;
            if (!find_cost_corr(u, v, c, used[c], edges_stack))
            {
                std::cout << u + 1 << " " << v + 1 << std::endl;
                std::cout << "cannot find route!" << std::endl;
                break;
            }

            for (auto &p : edges_stack)
            {
                if (p.second.first == c)
                {
                    edges[p.second.second].weight += p.first * cor.flow;
                    f += p.first * edges[p.second.second].cost;
                    // std::cout << p.first << " | " << edges[p.second.second].u + 1 << "->" << edges[p.second.second].v + 1 << std::endl;
                }
                else
                {
                    if (p.first > 0)
                    {
                        queue.push_back({edges[p.second.second].u, edges[p.second.second].v, p.second.first});
                    }
                    else
                    {
                        queue.push_back({edges[p.second.second].v, edges[p.second.second].u, p.second.first});
                    }
                }
            }
            queue.pop_front();
        }
        costs.push_back(f);
        sum += f * cor.flow;
    }
    std::vector<int> braess;
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].weight < 0)
            braess.push_back(i);
    }
    for (auto e : braess)
    {
        std::cout << e << ", ";
    }
    std::cout << std::endl;
    std::cout << sum << std::endl;
    return py::make_tuple(braess, costs, sum);
}

Graph::SumTree::SumTree(const std::vector<std::vector<int>> &neighb) : parents(neighb.size()),
                                                                       h(neighb.size(), -1),
                                                                       first(neighb.size(), -1),
                                                                       last(neighb.size(), -1),
                                                                       f(neighb.size() * 2 - 1),
                                                                       cost(neighb.size())
{
    std::vector<int> sorted;
    sorted.reserve(neighb.size() * 2 - 1);
    for (int i = 0; i < neighb.size(); i++)
    {
        if (neighb[i].size() > 0)
        {
            int c = 0;
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
    for (auto &v : neighb[u])
    {
        if (h[v] == -1)
        {
            parents[v] = u;
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

double Graph::SumTree::get(int u)
{
    return get_all(last[u]) - get_all(first[u] - 1);
}

PYBIND11_MODULE(graph_flow, m)
{
    py::class_<Graph>(m, "Graph")
        .def(py::init<int, double, std::vector<std::tuple<int, int, double, double>>>(),
             py::arg("n"), py::arg("eps") = 1e-9,
             py::arg("edges") = std::vector<std::tuple<int, int, double, double>>())
        .def("add_edge", &Graph::add_edge)
        .def("add_cost", &Graph::add_cost)
        .def("block_edge", &Graph::block_edge)
        .def("reuse_edge", &Graph::reuse_edge)
        .def("reset_flow", &Graph::reset_flow)
        .def("set_flow", &Graph::set_flow)
        .def("calculate_flow", &Graph::calculate_flow)
        .def("calculate_time", &Graph::calculate_time);
}
