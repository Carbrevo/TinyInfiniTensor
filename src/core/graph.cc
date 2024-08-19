#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <functional>
#include "core/runtime.h"
#include "core/optimizer.h"

using std::function;
using std::iterator;

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    void GraphObj::discnctOperatorAndRemove(const Operator &op)
    {
        IT_ASSERT(op->getSuccessors().size() == 1);
        auto opNext = op->getSuccessors()[0];

        sorted = false;
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(opNext);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(opNext);
                    opNext->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
        ops.erase(std::find(ops.cbegin(), ops.cend(), op));
    }

    void GraphObj::shortcutOperatorLink(const Operator &from, const Operator &to)
    {
        sorted = false;
        
        for (auto &input : from->getInputs())
        {
            IT_ASSERT(input);

            input->addTarget(to);
            //input->removeTarget(from);

            if (auto pred = input->getSource())
            {
                pred->addSuccessors(to);
                to->addPredecessors(pred);

                //pred->removeSuccessors(from);
                //from->removePredecessors(pred);
            }
        }

        auto prev = from;
        auto& toInput = to->getInputs();
        while (prev && std::find(toInput.cbegin(), toInput.cend(), prev->getOutput()) == toInput.cend()) {
            IT_ASSERT(prev->getSuccessors().size() == 1, "Not Implemented");
            prev = prev->getSuccessors()[0];
        }

        IT_ASSERT(prev);
        for (auto& output : prev->getOutputs()) {
            IT_ASSERT(output);

            output->removeTarget(to);

            if (auto pred = output->getSource())
            {
                pred->removeSuccessors(to);
                to->removePredecessors(pred);
            }

        }
        to->replaceInput(prev->getOutput(), from->getInputs()[0]);
    }

    void GraphObj::eliminateOperNode(const Operator &op)
    {
        sorted = false;

        for (auto &input : op->getInputs())
        {
            IT_ASSERT(input, "Invalid input");

            input->removeTarget(op);

            if (auto pred = input->getSource())
            {
                pred->removeSuccessors(op);
                op->removePredecessors(pred);
            }
        }

        for (auto &output : op->getOutputs())
        {
            IT_ASSERT(output, "Invalid output");

            auto test = Operator {};
            IT_ASSERT(!test);
            output->setSource(Operator{});
            for (auto &sussr : output->getTargets())
            {
                sussr->removePredecessors(op);
                op->removeSuccessors(sussr);
            }
        }

        ops.erase(std::find(ops.cbegin(), ops.cend(), op));
    }

    void GraphObj::skimOffTensors()
    {
        for( auto iter = tensors.begin(); iter != tensors.end();)
        {
            auto tensor = *iter;

            if (tensor->getSource()
                || tensor->getTargets().size() > 0 ) {
                iter++;
                continue;
            }

            std::cout << "Deleting tensor: " << tensor << std::endl;

            for (auto &target : tensor->getTargets()) {
                target->removeInput(tensor);
            }

            iter = tensors.erase(iter);
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        OptimizeContext optCtxt = make_ref<DFSOptContextObj>(std::dynamic_pointer_cast<GraphObj>(shared_from_this()));
        optCtxt->initOptimizers();

        while(!optCtxt->finished()) {
            while(!optCtxt->optimize());
            optCtxt->pushForward();
            std::cout << "\nafter a round-----\n" << optCtxt << std::endl;
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        //std::cout << __func__ << "() begin: num_tensors=" << tensors.size() << std::endl;
        //std::for_each(tensors.cbegin(), tensors.cend(), [](const auto &t){std::cout << "Tensor: " << t << std::endl;});

        void *primePtr = nullptr;
        auto iter = tensors.begin();
        auto end = tensors.end();
        function<void()> lmda_alloc = [this, &iter, &end, &primePtr, &lmda_alloc]{
            if (iter == end) {
                primePtr = allocator.getPtr();
                //std::cout << __func__ << "() prime_ptr=" << primePtr << std::endl;
                return;
            } else {
                auto curiter = iter++;
                auto shape = (*curiter)->getDims();
                auto offset = allocator.alloc((*curiter)->getBytes());            
                lmda_alloc();
                (*curiter)->setDataBlob(make_ref<BlobObj>(
                                        runtime,
                                        static_cast<uint8_t *>(primePtr) + offset));
                //std::cout << __func__ << "() " << "shape(";
                //std::for_each(shape.cbegin(), shape.cend(), [](const auto &d){std::cout << d << ",";});
                /*std::cout << ") at " << primePtr << " " 
                                << "+ " << offset
                                << " = " << (*curiter)->getBytes() << std::endl;
                                */
            }
        };

        lmda_alloc();

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {  
        for (auto tensor : tensors)
        {
            std::cout << "validate tensor " << tensor << std::endl;

            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()), "11111111");
            for (auto op : tensor->getTargets())
            {
                std::cout << "validate tensor target: " << op << std::endl;
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end(), "222222");
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()), "333333");
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end(), "4444444444");
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end(), "55555555555");
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end(), "6666666666");
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end(), "777777777");
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini