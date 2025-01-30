#include "autograd.h"
#include "Tensor.h"
#include <memory>
#include <optional>
#include <set>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include "TensorImpl.h"
#include "errors.h"

namespace statkitcpp {

void GraphTopSort(const std::shared_ptr<Node>& node,
                  std::vector<std::shared_ptr<Node>>& topsort,
                  std::unordered_set<std::shared_ptr<Node>>& used) {
    used.insert(node);
    if (!node->impl.lock()->IsLeaf()) {
        auto func = node->impl.lock()->grad_fn;
        for (const auto& v : func->GetChildren()) {
            if (!v->GetRequiresGrad()) {
                continue;
            }
            auto child_node = v->GetAutogradNode();
            if (v != nullptr && !used.contains(child_node)) {
                GraphTopSort(child_node, topsort, used);
            }
        }
    }
    topsort.push_back(node);
}

void RunBackward(const std::shared_ptr<Node>& root, std::optional<Tensor> grad_output, bool retain_graph) {
    //return;
    std::vector<std::shared_ptr<Node>> topsort;
    std::unordered_set<std::shared_ptr<Node>> used;
    
    GraphTopSort(root, topsort, used);
    if (grad_output.has_value()) {
        //std::cout << "Add grad_output for root " << root << std::endl;
        root->AddGrad(grad_output.value());
    }
    //return;
    while (!topsort.empty()) {
        auto node = topsort.back();
        topsort.pop_back();
        if (node->impl.lock()->GetRequiresGrad()) {
            if (node->impl.lock()->IsLeaf()) {
                node->impl.lock()->AddGrad(*node->temp_grad);
            } else if (retain_graph) {
                node->impl.lock()->AddGrad(*node->temp_grad);
            }
            // std::cout << node << ' ' << node->temp_grad << std::endl;
            if (!node->impl.lock()->IsLeaf()) {
                auto input_grads = node->impl.lock()->grad_fn->Backward(*node->temp_grad);
                auto inputs = node->impl.lock()->grad_fn->GetChildren();
                if (inputs.size() != input_grads.size()) {
                    throw InputsAndGradMismatchError{inputs.size(), input_grads.size()};
                }
                for (size_t idx = 0; idx < inputs.size(); idx++) {
                    if (input_grads[idx].GetImpl() != nullptr) {
                        auto child_node = inputs[idx]->GetAutogradNode();
                        // std::cout << node << "->" << child_node << std::endl;
                        child_node->AddGrad(input_grads[idx]);
                    }
                }
            }
        }
    }
}

}