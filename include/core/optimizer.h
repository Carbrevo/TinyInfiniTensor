#pragma once
#include "core/allocator.h"
#include "core/operator.h"
#include "core/tensor.h"

namespace infini
{
    class OperNoneObj : public OperatorObj
    {
    public:
        OperNoneObj() : OperatorObj(OpType::Unknown, {}, {}) {}
        OP_CLONE(OperNoneObj);
        optional<vector<Shape>> inferShape(const TensorVec &inputs) override
        {
            return std::nullopt;
        }

        std::string toString() const override
        {
            std::ostringstream os;

            os << "<OperNoneObj>";

            return os.str();
        }
        int numInputs() const override { return 0; }
        int numOutputs() const override { return 0; }
    };

    class OptimizeContextObj : public Object
    {
    using OptVec = vector<Optimizer>;
    protected:
        Graph m_graph;
        OptVec m_optRank;

    public:
        explicit OptimizeContextObj(Graph _graph)
            : m_graph(_graph){};

        virtual void    initOptimizers() = 0;
        virtual bool    finished() const { return true; }
        virtual void    pushForward() = 0;
                bool    optimize();

    public:
        void removeOperator(Operator op)
        {
            m_graph->removeOperator(op);
        }

        void removeTensor(Tensor tensor)
        {
            m_graph->removeTensor(tensor);
        }

        OpVec   searchOps(std::function<bool(const Operator&)> inspector) const
        {
            auto opList = m_graph->ops;
            OpVec   opsFound;

            std::for_each(opList.cbegin(), opList.cend(), 
                            [&opsFound, &inspector](const auto& op) {
                                if ( inspector(op) ) {
                                    opsFound.emplace_back( op );
                                }
                            });

            return opsFound;
        }

        void shortcutOperatorLink(const Operator &from, const Operator &to)
        {
            m_graph->shortcutOperatorLink(from, to);            
        }

        void eliminateOperNode(const Operator &op)
        {
            m_graph->eliminateOperNode(op);            
        }

        void skimOffTensors()
        {
            m_graph->skimOffTensors();
        }
    };

    class OptimizerObj : public Object
    {
    protected:

    public:
        explicit OptimizerObj() {};

    public:
        virtual bool    optimize() { return true; }        
    };

    class DFSOptContextObj : public OptimizeContextObj
    {
    protected:
        vector<OpVec> m_pathQueue;

    public:
        explicit DFSOptContextObj(Graph _graph)
            : OptimizeContextObj(_graph){
            auto opsRoot = searchOps([](const auto& op) { return op->getPredecessors().size() == 0;});
            std::for_each(opsRoot.cbegin(), opsRoot.cend(),
                            [this](const auto& op) { m_pathQueue.emplace_back(OpVec {op});});
        }     

        string toString() const override
        {
            std::ostringstream oss;
            int idx = 0;

            oss << "DFSOptContext:\n";
            for (const auto &ops : m_pathQueue)
                oss << "[" << idx++ << "] "
                    <<vecToString(ops) << "\n";

            oss << "Optimizer: " << vecToString(m_optRank);

            return oss.str();
        }

    public:
        OpVec&  getCurPath()
        {
            return *std::find_if(m_pathQueue.begin(), m_pathQueue.end(),
                        [](const OpVec& ops) -> bool {Operator opLast = *ops.crbegin(); return opLast->getOpType() != OpType::Unknown;});
        }

        void shortcutOperatorLink(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd)
        {
            auto from = *(rEnd - 1);
            auto last = *rFirst;
            IT_ASSERT(last->getSuccessors().size() == 1);
            auto to = last->getSuccessors()[0];

            OptimizeContextObj::shortcutOperatorLink(from, to);

            std::cout<< "cut from " << from << " to " << to << std::endl;
            auto& pathRoot = getCurPath();
            auto itDelete = std::find(pathRoot.begin(), pathRoot.end(), from);
            while (itDelete != pathRoot.end()) {
                OptimizeContextObj::eliminateOperNode(*itDelete);
                std::cout << "Deleting " << *itDelete << std::endl;
                itDelete = pathRoot.erase(itDelete);
            }
            OptimizeContextObj::skimOffTensors();
            IT_ASSERT(m_graph->checkValid());

            std::cout<< "add check " << to << std::endl;
            pathRoot.emplace_back(to);
            std::cout<< __func__ << " complete." << to << std::endl;
        }

        void mergeOperatorLink(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd)
        {
            auto from = *(rEnd - 1);
            auto to = *rFirst;

            OptimizeContextObj::shortcutOperatorLink(from, to);

            std::cout<< "merge from " << from << " into " << to << std::endl;
            auto& pathRoot = getCurPath();
            auto itDelete = std::find(pathRoot.begin(), pathRoot.end(), from);
            while (itDelete != pathRoot.end()) {
                if (*itDelete == to) {
                    break;
                }
                OptimizeContextObj::eliminateOperNode(*itDelete);
                std::cout << "Deleting " << *itDelete << std::endl;
                itDelete = pathRoot.erase(itDelete);
            }
            OptimizeContextObj::skimOffTensors();
            IT_ASSERT(m_graph->checkValid());
        }

    public:
        virtual void    initOptimizers() override;
        virtual bool    finished() const override
        {
            return std::all_of(m_pathQueue.cbegin(), m_pathQueue.cend(),
                                [](const auto& ops) { auto opLast = *ops.crbegin(); return opLast->getOpType() == OpType::Unknown;}); 
        }

        virtual void    pushForward() override
        {
            auto& pathCheck = getCurPath();
            auto opCheck = *pathCheck.crbegin();

            if(opCheck->getSuccessors().empty()) {
                opCheck = make_ref<OperNoneObj>();
                pathCheck.emplace_back(opCheck);
            } else {
                auto susrs = opCheck->getSuccessors();
                auto susrsIter = susrs.cbegin();
                opCheck = *susrsIter++;

                std::for_each(susrsIter, susrs.cend(),
                            [this, newPath = pathCheck](auto &op)mutable{
                                newPath.emplace_back( op );
                                m_pathQueue.emplace_back(newPath);
                            });
                pathCheck.emplace_back(opCheck);
            }
        }
    };

    using   DFSOptContext = Ref<DFSOptContextObj>;
    class DFSOptNoneObj : public OptimizerObj
    {
    protected:
        DFSOptContext m_optCtxt;

    public:
        explicit DFSOptNoneObj(DFSOptContext _optCtxt)
            : m_optCtxt(_optCtxt){};
        
        string toString() const override
        {
            std::ostringstream oss;

            oss << "DFSOptNone";

            return oss.str();
        }

    public:
        virtual bool    optimize() override;

    public:
        bool    optMerge(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd);
    };

    class DFSOptTransposeObj : public OptimizerObj
    {
    protected:
        DFSOptContext m_optCtxt;

    public:
        explicit DFSOptTransposeObj(DFSOptContext _optCtxt)
            : m_optCtxt(_optCtxt){};
        
        string toString() const override
        {
            std::ostringstream oss;

            oss << "DFSOptTranspose";

            return oss.str();
        }

    public:
        virtual bool    optimize() override;

    public:
        bool    optMerge(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd);
    };

    class DFSOptTransMatmultObj : public OptimizerObj
    {
    protected:
        DFSOptContext m_optCtxt;

    public:
        explicit DFSOptTransMatmultObj(DFSOptContext _optCtxt)
            : m_optCtxt(_optCtxt){};
        
        string toString() const override
        {
            std::ostringstream oss;

            oss << "DFSOptTransMatmult";

            return oss.str();
        }

    public:
        virtual bool    optimize() override;

    public:
        bool    optMerge(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd);
    };
}