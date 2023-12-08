#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lp_lib.h"

namespace py = pybind11;

#include <set>
#include <vector>
#include <limits>
#include <iostream>

const double INF = std::numeric_limits<double>::max();
const double EPS = 1e-9;

class edge
{
public:
    int u, v;
    double flow = 0, max_flow;
    double time = -1, cost;
    bool free = false;
    bool blocked = false;
    double weight = 0;

    edge(int u, int v, double max_flow, double cost);
    double get_residual_capacity() const;

    void add_flow(double f);
};

edge::edge(int u, int v, double max_flow, double cost) : u(u),
                                                         v(v),
                                                         max_flow(max_flow),
                                                         cost(cost)
{
}

inline double edge::get_residual_capacity() const
{
    return max_flow - flow;
}

inline void edge::add_flow(double f)
{
    flow += f;
}

class Graph
{
public:
    std::vector<edge> edges;
    std::vector<std::vector<int>> vert;
    Graph(int n, const std::vector<std::tuple<int, int, double, double>> &edges)
    {
        vert = std::vector<std::vector<int>>(n, std::vector<int>(0));
        d_vertices = std::vector<std::vector<int>>(n, std::vector<int>(0));
        d_flow = std::vector<std::vector<double>>(n, std::vector<double>(0));

        for (auto &e : edges)
        {
            const auto [u, v, max_flow, cost] = e;
            this->add_edge(u, v, max_flow, cost);
        }
    }
    int add_edge(int u, int v, double max_flow, double cost);
    void block_edge(int id);
    void reuse_edge(int id);
    void set_flow(const std::vector<std::tuple<int, int, double>> &flow);
    py::tuple calculate_flow();
    void reset_flow();
    void delete_flow();
    std::vector<int> find_br();

private:
    bool flow_linear();
    std::vector<double> potential;
    std::vector<std::vector<int>> d_vertices;
    std::vector<std::vector<double>> d_flow;
    bool run_flow(int u, int v, double f);
    double add_flow(int u, int v, double cap);
    void dfs(int u, std::set<int> &used, bool backward);
    void dfs_residual(int u, std::vector<bool> &used, std::vector<float> &p, std::set<int> &possible);
    void dfs_br(int u, std::vector<bool> &used, std::vector<double> &end_path);
};

int Graph::add_edge(int u, int v, double max_flow, double cost)
{
    vert[u].push_back(edges.size());
    vert[v].push_back(edges.size() + 1);
    edges.emplace_back(u, v, max_flow, cost);
    edges.emplace_back(v, u, 0, -cost);
    return (edges.size() / 2) - 1;
}

void Graph::block_edge(int id)
{
    edges[id * 2].blocked = true;
    edges[id * 2 + 1].blocked = true;
}

void Graph::reuse_edge(int id)
{
    edges[id * 2].blocked = false;
    edges[id * 2 + 1].blocked = false;
}

void Graph::reset_flow()
{
    for (auto &e : edges)
    {
        e.flow = 0;
        e.free = false;
        e.time = -1;
        e.weight = 0;
    }
}

void Graph::set_flow(const std::vector<std::tuple<int, int, double>> &flow)
{
    for (auto &s : flow)
    {
        auto [u, v, f] = s;
        d_vertices[u].push_back(v);
        d_flow[u].push_back(f);
    }
}

void Graph::delete_flow()
{
    reset_flow();
    for (auto &d : d_vertices)
    {
        d.clear();
    }
    for (auto &d : d_flow)
    {
        d.clear();
    }
}

bool Graph::flow_linear()
{
    int edge_cols = 0;
    for (auto &v : d_vertices)
    {
        edge_cols += v.size();
    }
    int n_cols = edge_cols * edges.size() / 2;
    lprec *lp = make_lp(0, n_cols);
    if (lp == NULL)
    {
        return false;
    }
    std::vector<std::vector<std::string>> labels(d_vertices.size());
    int cnt = 0;
    for (int k = 0; k < edges.size() / 2; k++)
    {
        for (int i = 0; i < d_vertices.size(); i++)
        {
            for (auto j : d_vertices[i])
            {
                cnt++;
                labels[i].push_back("(" + std::to_string(i) + "->" + std::to_string(j) + "_" + std::to_string(k) + ")");
                set_col_name(lp, cnt, labels[i].back().data());
            }
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
    set_add_rowmode(lp, TRUE);
    REAL one = 1;
    for (int k = 0; k < edges.size() / 2; k++)
    {
        for (int i = 0; i < d_vertices.size(); i++)
        {
            for (auto j : d_vertices[i])
            {
                cnt++;
                if (edges[2 * k].blocked)
                {
                    if (!add_constraintex(lp, 1, &one, &cnt, EQ, 0))
                    {
                        free(col_no);
                        free(row);
                        delete_lp(lp);
                        return false;
                    }
                }
            }
        }
    }
    for (int k = 0; k < edges.size() / 2; k++)
    {
        for (int i = 0; i < edge_cols; i++)
        {
            col_no[i] = k * edge_cols + i + 1;
        }
        if (!add_constraintex(lp, edge_cols, row, col_no, LE, edges[k * 2].max_flow))
        {
            free(col_no);
            free(row);
            delete_lp(lp);
            return false;
        }
    }
    free(col_no);
    free(row);
    col_no = (int *)calloc(edges.size() / 2, sizeof(int));
    row = (REAL *)calloc(edges.size() / 2, sizeof(REAL));
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
    for (int i = 0; i < d_vertices.size(); i++)
    {
        for (int j = 0; j < d_vertices[i].size(); j++)
        {
            for (int k = 0; k < vert.size(); k++)
            {
                for (int t = 0; t < edges.size() / 2; t++)
                {
                    col_no[t] = 0;
                    row[t] = 0;
                }
                for (int t = 0; t < vert[k].size(); t++)
                {
                    col_no[t] = (vert[k][t] / 2) * edge_cols + cnt + 1;
                    if (vert[k][t] % 2 == 0)
                    {
                        row[t] = 1.;
                    }
                    else
                    {
                        row[t] = -1.;
                    }
                }
                double sum = 0;
                if (i != d_vertices[i][j])
                {
                    if (i == k)
                    {
                        sum = d_flow[i][j];
                    }
                    if (d_vertices[i][j] == k)
                    {
                        sum = -d_flow[i][j];
                    }
                }
                if (!add_constraintex(lp, vert[k].size(), row, col_no, EQ, sum))
                {
                    free(col_no);
                    free(row);
                    delete_lp(lp);
                    return false;
                }
            }
            cnt++;
        }
    }
    set_add_rowmode(lp, FALSE);
    free(col_no);
    free(row);
    col_no = (int *)calloc(n_cols, sizeof(int));
    row = (REAL *)calloc(n_cols, sizeof(REAL));
    cnt = 0;
    for (int k = 0; k < edges.size() / 2; k++)
    {
        for (int i = 0; i < d_vertices.size(); i++)
        {
            for (auto j : d_vertices[i])
            {
                row[cnt] = edges[k * 2].cost;
                col_no[cnt] = cnt + 1;
                cnt++;
            }
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
    if (solve(lp) != OPTIMAL)
    {
        free(col_no);
        free(row);
        delete_lp(lp);
        return false;
    }

    // variable values
    get_variables(lp, row);
    cnt = 0;
    for (int k = 0; k < edges.size() / 2; k++)
    {
        edges[k].flow = 0;
    }
    for (int k = 0; k < edges.size() / 2; k++)
    {
        for (int i = 0; i < d_vertices.size(); i++)
        {
            for (auto j : d_vertices[i])
            {
                edges[k * 2].flow += row[cnt];
                edges[k * 2 + 1].flow -= row[cnt];
                cnt++;
            }
        }
    }
    free(col_no);
    free(row);
    delete_lp(lp);
    return true;
}

bool Graph::run_flow(int u, int v, double f)
{
    potential = std::vector<double>(vert.size());
    while (true)
    {
        double flow = add_flow(u, v, f);
        if (std::abs(flow - f) < EPS)
        {
            return true;
        }
        if (flow < EPS)
        {
            return false;
        }
        f -= flow;
    }
}

double Graph::add_flow(int u, int v, double cap)
{
    std::vector<bool> vis(vert.size(), false);
    std::vector<double> dist(vert.size(), INF);
    dist[u] = 0;
    std::vector<int> parent(vert.size(), -1);

    std::set<std::pair<double, int>> min_dist{{0., u}};

    auto hint = min_dist.begin();
    for (int i = 0; i < vert.size(); ++i)
        if (i != u)
        {
            hint = min_dist.insert(hint, {INF, i});
        }

    while (!min_dist.empty())
    {
        std::pair<double, int> p = *min_dist.begin();
        auto [d, u1] = p;

        if (d == INF)
        {
            if (u1 == v)
            {
                return 0;
            }
            break;
        }
        min_dist.erase(min_dist.begin());
        vis[u1] = true;

        for (const int &e : vert[u1])
        {
            edge &v_e = edges[e];
            if (!v_e.blocked)
            {
                if (v_e.get_residual_capacity() > EPS)
                {
                    double new_d = d + v_e.cost + potential[u1] - potential[v_e.v];
                    if (!vis[v_e.v] && dist[v_e.v] > new_d)
                    {
                        min_dist.erase({dist[v_e.v], v_e.v});
                        min_dist.insert({new_d, v_e.v});
                        dist[v_e.v] = new_d;
                        parent[v_e.v] = e;
                    }
                }
            }
        }
    }

    if (dist[v] == INF)
    {
        return 0;
    }
    for (auto i = 0; i < dist.size(); i++)
    {
        dist[i] += potential[i] - potential[u];
    }
    dist.swap(potential);
    int vertex = v;
    double flow = INF;
    while (vertex != u)
    {
        if (flow > edges[parent[vertex]].get_residual_capacity())
        {
            flow = edges[parent[vertex]].get_residual_capacity();
        };
        vertex = edges[parent[vertex]].u;
    }
    if (flow > cap)
    {
        flow = cap;
    }
    vertex = v;
    while (vertex != u)
    {
        edges[parent[vertex] ^ 1].add_flow(-flow);
        edges[parent[vertex]].add_flow(flow);
        vertex = edges[parent[vertex]].u;
    }
    return flow;
}

void Graph::dfs(int u, std::set<int> &used, bool backward)
{
    for (auto i : vert[u])
    {
        if (!edges[i].blocked)
        {
            int v = edges[i].v;
            if (((!backward && (i % 2 == 0 && edges[i].flow > EPS)) ||
                 (backward && (i % 2 == 1 && edges[i].flow < -EPS))) &&
                (used.find(v) == used.end()))
            {
                int v = edges[i].v;
                used.insert(v);
                dfs(v, used, backward);
            }
        }
    }
}

void Graph::dfs_residual(int u, std::vector<bool> &used, std::vector<float> &p, std::set<int> &possible)
{
    for (auto i : vert[u])
    {
        if (!edges[i].blocked)
        {
            if (i % 2 == 0 && edges[i].flow < edges[i].max_flow && possible.find(edges[i].v) != possible.end() && !used[edges[i].v])
            {
                edges[i].free = true;
                edges[i].time = edges[i].cost;
                p[edges[i].v] = p[u] + edges[i].cost;
                used[edges[i].v] = true;
                dfs_residual(edges[i].v, used, p, possible);
            }
            else if (i % 2 == 1 && edges[i ^ 1].flow < edges[i ^ 1].max_flow && possible.find(edges[i].v) != possible.end() && !used[edges[i].v])
            {
                edges[i ^ 1].free = true;
                edges[i ^ 1].time = edges[i ^ 1].cost;
                p[edges[i].v] = p[u] + edges[i].cost;
                used[edges[i].v] = true;
                dfs_residual(edges[i].v, used, p, possible);
            }
        }
    }
}

py::tuple Graph::calculate_flow()
{
    bool clear = true;
    if (!flow_linear())
    {
        return py::make_tuple(false, std::vector<py::tuple>(0), std::vector<py::tuple>(0));
    }
    for (auto &e : edges)
    {
        std::cout << e.flow << " ";
    }
    std::cout << std::endl;
    std::vector<std::set<int>> connected_sets;
    std::vector<std::vector<double>> time(d_vertices.size(), std::vector<double>());
    for (int i = 0; i < d_vertices.size(); i++)
    {
        if (d_vertices[i].size() > 0)
        {
            std::set<int> used_u;
            dfs(i, used_u, false);
            for (auto j : d_vertices[i])
            {
                std::set<int> used_v;
                dfs(j, used_v, true);
                used_u.merge(used_v);
            }

            std::vector<float> p(vert.size(), -1);
            p[i] = 0;
            std::vector<bool> used(vert.size(), false);
            dfs_residual(i, used, p, used_u);
            for (auto d_v : d_vertices[i])
            {
                time[i].push_back(p[d_v]);
            }
            for (int i = 0; i < edges.size(); i += 2)
            {
                if (!edges[i].blocked)
                {
                    if (p[edges[i].u] >= 0 && p[edges[i].v] >= 0)
                    {
                        double d = p[edges[i].v] - p[edges[i].u];
                        if (edges[i].cost < d)
                        {
                            edges[i].time = d;
                            edges[i].free = false;
                        }
                    }
                }
            }
        }
    }

    bool solved = true;
    std::vector<py::tuple> result_time;
    for (int i = 0; i < d_vertices.size(); i++)
    {
        for (int j = 0; j < d_vertices[i].size(); j++)
        {
            if (time[i][j] < 0)
            {
                solved = false;
            }
            result_time.push_back(py::make_tuple(i, d_vertices[i][j], time[i][j]));
        }
    }

    std::vector<py::tuple> result_flow;
    for (int i = 0; i < edges.size(); i += 2)
    {
        result_flow.push_back(py::make_tuple(edges[i].flow, edges[i].time));
    }

    return py::make_tuple(solved, result_time, result_flow);
}

void Graph::dfs_br(int u, std::vector<bool> &used, std::vector<double> &end_path)
{
    used[u] = true;
    for (auto i : vert[u])
    {
        if (((i % 2 == 0 && edges[i].free) ||
             (i % 2 == 1 && edges[i ^ 1].free)) &&
            !edges[i].blocked)
        {
            if (!used[edges[i].v])
            {
                dfs_br(edges[i].v, used, end_path);
            }
            if (end_path[edges[i].v] != 0)
            {
                if (i % 2 == 1)
                {
                    edges[i ^ 1].weight -= end_path[edges[i].v];
                }
                else
                {
                    edges[i].weight += end_path[edges[i].v];
                }
                end_path[u] = end_path[edges[i].v];
            }
        }
    }
}

std::vector<int> Graph::find_br()
{
    std::vector<int> res;
    for (int i = 0; i < d_vertices.size(); i++)
    {
        std::vector<bool> used(vert.size(), false);
        std::vector<double> weights(vert.size(), 0);
        for (int j = 0; j < d_vertices[i].size(); j++)
        {
            weights[d_vertices[i][j]] = d_flow[i][j];
        }
        dfs_br(i, used, weights);
    }
    for (int i = 0; i < edges.size(); i += 2)
    {
        if (edges[i].weight < 0 && !edges[i].blocked)
        {
            res.push_back(i / 2);
        }
    }
    return res;
}

PYBIND11_MODULE(graph_flow, m)
{
    py::class_<Graph>(m, "Graph")
        .def(py::init<int, const std::vector<std::tuple<int, int, double, double>> &>(),
             py::arg("n"), py::arg("edges") = std::vector<std::tuple<int, int, double, double>>())
        .def("add_edge", &Graph::add_edge)
        .def("block_edge", &Graph::block_edge)
        .def("reuse_edge", &Graph::reuse_edge)
        .def("reset_flow", &Graph::reset_flow)
        .def("delete_flow", &Graph::delete_flow)
        .def("set_flow", &Graph::set_flow)
        .def("calculate_flow", &Graph::calculate_flow)
        .def("find_braess", &Graph::find_br);
}
