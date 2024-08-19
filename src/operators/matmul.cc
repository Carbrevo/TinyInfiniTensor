#include "operators/matmul.h"
#include "utils/operator_utils.h"

using std::cout;
using std::cerr;
using std::endl;

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul[" << getGuid() << "]([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";

        //IT_ASSERT(inputs.size() == 2, std::string("size=") + std::to_string(inputs.size()));

        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);
        //std::cout<<__func__<<": num_tensor="<<inputs.size()<<std::endl;
        //std::for_each(inputs.cbegin(), inputs.cend(), [](const auto &t){std::cout << "Tensor: " << t << std::endl;});

        auto A = inputs[0], B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        int rankA = A->getRank(); // Rank is the Shape of TensorDims
        int rankB = B->getRank();
        Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
        Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
        Shape inferShapeAB = infer_broadcast(shapeA1, shapeB1);
        auto riterA = shapeA.crbegin();
        auto riterB = shapeB.crbegin();
        auto [kA, mA] = transA ? std::make_pair( *(riterA + 1), *riterA) : std::make_pair( *riterA, *(riterA + 1));
        auto [kB, mB] = transB ? std::make_pair( *riterB, *(riterB + 1)) : std::make_pair( *(riterB + 1), *riterB);
        IT_ASSERT(kA == kB);
        m = mA;
        n = mB;
        k = kA;
        inferShapeAB.emplace_back(m);
        inferShapeAB.emplace_back(n);
        return {{inferShapeAB}};
    }
} // namespace infini