#include <algorithm>
#include <numeric>
#include <queue>
#include <format>
#include <functional>
#include "core/graph.h"
#include "core/runtime.h"
#include "core/optimizer.h"
#include "operators/transpose.h"
#include "operators/matmul.h"

using std::function;
using std::iterator;

namespace infini
{
    using OperNone = Ref<OperNoneObj>;
    using OperTranspose = Ref<TransposeObj>;
    using OperMatmul = Ref<MatmulObj>;

    bool    OptimizeContextObj::optimize()
    {
        IT_ASSERT(!finished());

        bool succ;
        do {
            succ = true;
            for (auto opt : m_optRank)
            {
                succ = succ && opt->optimize();

            }
        } while (!succ);

        return succ;
    }

    void    DFSOptContextObj::initOptimizers()
    {
        m_optRank.emplace_back(make_ref<DFSOptTransposeObj>(std::dynamic_pointer_cast<DFSOptContextObj>(shared_from_this())));
        m_optRank.emplace_back(make_ref<DFSOptTransMatmultObj>(std::dynamic_pointer_cast<DFSOptContextObj>(shared_from_this())));
    }
    bool    DFSOptTransposeObj::optMerge(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd)
    {
        std::cout << "Doing " << toString() << ": " <<__func__ << std::endl;
        std::cout << *rFirst << std::endl;

        auto origin = std::dynamic_pointer_cast<TransposeObj>(*rFirst)->getPermute();
        std::for_each(origin.begin(), origin.end(), [idx = 0](auto& x) mutable { x = idx++;} );
        auto mergen = origin;
        for (auto iter = rEnd - 1; iter >= rFirst; iter--) {
            OperTranspose op = std::dynamic_pointer_cast<TransposeObj>(*iter);

            auto permute = op->getPermute();
            auto input = mergen;
            for (int i = 0; i < static_cast<int>(permute.size()); ++i) {
                mergen[i] = input[permute[i]];
            }
        }

        if ( mergen == origin ) {
            std::cout << "Redundant" << std::endl;

            m_optCtxt->shortcutOperatorLink(rFirst, rEnd);
            std::cout << "optimize-----" << m_optCtxt << std::endl;
            return false;
        } else {
            std::cout << "Mergen: " << vecToString(mergen) << std::endl;
        }
        return true;
    }

    bool    DFSOptTransposeObj::optimize()
    {
        auto& pathCheck = m_optCtxt->getCurPath();
        auto iterRecall = pathCheck.rbegin();
        auto opCheck = *iterRecall;

        if (opCheck->getOpType() ==  OpType::Transpose
                && opCheck->getPredecessors().size() == 1
                && opCheck->getSuccessors().size() == 1) {
            auto iterFirst = iterRecall;

            while(++iterRecall != pathCheck.rend()) {
                auto opPrevNCheck = *iterRecall;

                if(opPrevNCheck->getOpType() !=  OpType::Transpose
                    || opPrevNCheck->getPredecessors().size() > 1
                    || opPrevNCheck->getSuccessors().size() != 1) {
                        break;
                }
            }

            auto num = iterRecall - iterFirst;
            std::cout << "Transpose Checking: " << num << std::endl;
            if (num >= 2) {
                return optMerge(iterFirst, iterRecall);
            }
        }
    
        return true; 
    }    

    bool    DFSOptTransMatmultObj::optMerge(OpVec::reverse_iterator rFirst, OpVec::reverse_iterator rEnd)
    {
        std::cout << "Doing " << toString() << ": " <<__func__ << std::endl;

        auto opTrans = *(rEnd - 1);
        OperMatmul opMatmul = std::dynamic_pointer_cast<MatmulObj>(*rFirst);
        std::cout << opTrans << std::endl;
        std::cout << opMatmul << std::endl;

        //IT_ASSERT(opMatmul->getInputs().size() == 2, 
          //          std::string("input size=") + to_string(opMatmul->getInputs().size()));
        auto inputA = opMatmul->getInputs()[0];
        auto inputB = opMatmul->getInputs()[1];
        if (opTrans == inputA->getSource()) {
            opMatmul->setTransA(true);
        } else if (opTrans == inputB->getSource()) {
            opMatmul->setTransB(true);
        } else {
            std::cout << m_optCtxt << std::endl;
            std::cout << inputA << std::endl;
            std::cout << inputB << std::endl;
            IT_ASSERT(false, "Invalid Graph");
        }

        m_optCtxt->mergeOperatorLink(rFirst, rEnd);

        std::cout << "optimize-----" << m_optCtxt << std::endl;
        return false;
    }
    
    bool    DFSOptTransMatmultObj::optimize()
    {
        auto& pathCheck = m_optCtxt->getCurPath();
        auto iterRecall = pathCheck.rbegin();
        auto opCheck = *iterRecall;

        if (opCheck->getOpType() ==  OpType::MatMul) {
            auto iterFirst = iterRecall;

            if (++iterRecall != pathCheck.rend()) {
                auto opPrevNCheck = *iterRecall;

                if(opPrevNCheck->getOpType() !=  OpType::Transpose
                    || opPrevNCheck->getPredecessors().size() > 1
                    || opPrevNCheck->getSuccessors().size() != 1) {
                        return true;
                }
                iterRecall++;

                OperTranspose opTrans = std::dynamic_pointer_cast<TransposeObj>(opPrevNCheck);
                auto permute = opTrans->getPermute();
                if (permute.size() < 2 ) {
                    return true;
                }

                auto m = permute.size() - 2;
                auto n = m + 1;

                if (permute[m] != static_cast<int>(n) 
                    || permute[n] != static_cast<int>(m)) {
                    return true;
                }

                return optMerge(iterFirst, iterRecall);
            }

        }
    
        return true; 
    }    
}