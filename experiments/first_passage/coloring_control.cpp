// Fixed-budget, one-thread TabuCol-style non-neural control for verified k-coloring.
// Input consists of "id vertices edges" followed by that many undirected edge pairs.
// Each line of output contains id, moves, internal elapsed seconds and the coloring on success.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Graph = std::vector<std::vector<int>>;
using Matrix = std::vector<std::vector<int>>;
using Clock = std::chrono::steady_clock;

std::vector<int> initial_coloring(const Graph& graph, Matrix& neighbor_counts) {
    const int n = static_cast<int>(graph.size());
    std::vector<int> values(n, -1), saturation(n, 0);
    for (int step = 0; step < n; ++step) {
        int best = -1;
        for (int i = 0; i < n; ++i) {
            if (values[i] >= 0) continue;
            if (best < 0 || saturation[i] > saturation[best] ||
                (saturation[i] == saturation[best] && graph[i].size() > graph[best].size())) best = i;
        }
        auto& counts = neighbor_counts[best];
        values[best] = static_cast<int>(std::min_element(counts.begin(), counts.end()) - counts.begin());
        for (int neighbor : graph[best]) {
            if (neighbor_counts[neighbor][values[best]]++ == 0) ++saturation[neighbor];
        }
    }
    return values;
}

int conflicts(const Graph& graph, const std::vector<int>& color) {
    int violations = 0;
    for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
        for (int neighbor : graph[i]) {
            if (neighbor > i && color[i] == color[neighbor]) ++violations;
        }
    }
    return violations;
}

void solve(const Graph& graph, int id, int colors, int max_moves, double max_seconds) {
    const auto start = Clock::now();
    const int n = static_cast<int>(graph.size());
    std::mt19937 rng(20261123u + static_cast<std::uint32_t>(id));
    Matrix counts(n, std::vector<int>(colors, 0)), tabu_until(n, std::vector<int>(colors, 0));
    auto values = initial_coloring(graph, counts);
    int cost = conflicts(graph, values), best_cost = cost, moves = 0;
    while (cost > 0 && moves < max_moves) {
        if (std::chrono::duration<double>(Clock::now() - start).count() >= max_seconds) break;
        int best_delta = std::numeric_limits<int>::max(), choice = -1, new_color = -1, ties = 0;
        for (int vertex = 0; vertex < n; ++vertex) {
            int old_cost = counts[vertex][values[vertex]];
            if (old_cost == 0) continue;
            for (int color = 0; color < colors; ++color) {
                if (color == values[vertex]) continue;
                int delta = counts[vertex][color] - old_cost;
                if (tabu_until[vertex][color] > moves && cost + delta >= best_cost) continue;
                if (delta > best_delta) continue;
                if (delta < best_delta) {
                    best_delta = delta;
                    ties = 0;
                }
                if (std::uniform_int_distribution<int>(1, ++ties)(rng) == 1) {
                    choice = vertex;
                    new_color = color;
                }
            }
        }
        // All conflicting moves tabu: release prohibitions and continue.
        if (choice < 0) {
            for (auto& row : tabu_until) std::fill(row.begin(), row.end(), 0);
            continue;
        }
        const int old_color = values[choice];
        tabu_until[choice][old_color] = moves + 1 + static_cast<int>(0.6 * cost) +
                                       std::uniform_int_distribution<int>(1, 10)(rng);
        for (int neighbor : graph[choice]) {
            --counts[neighbor][old_color];
            ++counts[neighbor][new_color];
        }
        values[choice] = new_color;
        cost += best_delta;
        best_cost = std::min(cost, best_cost);
        ++moves;
    }
    if (cost != conflicts(graph, values)) throw std::logic_error("Incremental cost disagrees");
    auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << id << ' ' << moves << ' ' << std::setprecision(9) << elapsed << ' ' << best_cost;
    if (cost == 0) for (int color : values) std::cout << ' ' << color;
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: coloring_control colors max_moves max_seconds < graphs.txt\n";
        return 2;
    }
    const int colors = std::stoi(argv[1]), limit = std::stoi(argv[2]);
    const double seconds = std::stod(argv[3]);
    if (colors < 2 || limit < 0 || !std::isfinite(seconds) || seconds <= 0.) return 2;
    int id, n, m;
    while (std::cin >> id >> n >> m) {
        if (n <= 0 || m < 0) return 2;
        Graph graph(n);
        for (int e = 0; e < m; ++e) {
            int u, v;
            if (!(std::cin >> u >> v) || u < 0 || u >= v || v >= n) return 2;
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
        solve(graph, id, colors, limit, seconds);
    }
    return std::cin.eof() ? 0 : 2;
}
