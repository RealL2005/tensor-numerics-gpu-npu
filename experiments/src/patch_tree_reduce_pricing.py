#!/usr/bin/env python3
"""给 TreeReduceV2 加一个"精度定价"诊断（emit-precision-report，默认关）。

只增加诊断输出，不改任何 lowering 决策/生成逻辑：
  - Passes.td 增一个 bool 选项 emit-precision-report
  - .cpp 增 emitPrecisionCost(plan)：打印 dimR/dimA/vecLen/stages/strategy 与两条上界
        A_max = (δ/u)·R ≤ R          （顺序累加器，R = 吞并次数）
        A_max ≤ ceil(log2 R)         （平衡树）
    以及 price(chain/tree) = R/log2(R)，即"选错策略最多贵多少倍"。
锚点全部断言，匹配不上直接退出，不做部分修改。
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "AscendNPU-IR"
TD = ROOT / "bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td"
CPP = ROOT / "bishengir/lib/Dialect/HFusion/Transforms/TreeReduceV2.cpp"

# ---------------------------------------------------------------- Passes.td
td_anchor = """    Option<"onlyLegacyScope", "only-legacy-scope", "bool", "false",
           "Rewrite only modules selected for the legacy tree strategy.">,
"""
td_add = """    Option<"emitPrecisionReport", "emit-precision-report", "bool", "false",
           "Print for every rewritten reduction the structural parameters "
           "(dimR/dimA/vectorLength/stages/strategy) and the precision bound "
           "they imply, in units of one single rounding.">,
"""

# ---------------------------------------------------------------- .cpp
decl_anchor = "  void runOnOperation() override;\n"
decl_add = "  void emitPrecisionCost(const ReductionRewritePlan &plan);\n"

inc_anchor = '#include "bishengir/Dialect/HACC/Utils/Utils.h"\n'
inc_add = inc_anchor + '#include "llvm/Support/raw_ostream.h"\n'

impl_anchor = """LogicalResult
TreeReduceV2Pass::executeRewrite(IRRewriter &rewriter,
                                 const ReductionRewritePlan &plan) {
  Location loc = plan.rewriteLoop->getLoc();
  rewriter.setInsertionPoint(plan.rewriteLoop);
"""
impl_add = '''/// Precision pricing.
///
/// The pass picks a *structure* per reduction (balanced register tree / staged
/// folding / in-place multi-pass folding / legacy sequential). Those structures
/// differ in how many times the accumulator is rounded while its addend is
/// still below half an ulp, which is exactly the quantity that bounds the
/// error:
///
///     A_max = (delta/u) * R,   delta <= ulp(acc)/2   =>   A_max <= R
///     A     = |err| / (u * sum|x|),                    u = 2^-24
///
/// so a sequential accumulator over dimR gives A_max <= dimR while a balanced
/// tree over the same dimR gives A_max <= ceil(log2 dimR): the same reduction,
/// the same hardware, up to dimR/log2(dimR) apart in worst-case error.
/// Printing both bounds here makes the price of the strategy choice visible in
/// the lowering stage instead of in a downstream tolerance test.
static int64_t treeDepthBound(int64_t n) {
  int64_t d = 0;
  int64_t v = 1;
  while (v < n) {
    v *= 2;
    ++d;
  }
  return d;
}

static const char *structureClass(ReductionLoweringKind kind) {
  switch (kind) {
  case ReductionLoweringKind::DirectRegisterRA:
    return "tree";
  case ReductionLoweringKind::LegacyRA:
    return "chain";
  case ReductionLoweringKind::AscendStyleRA:
  case ReductionLoweringKind::AscendStyleAR:
  case ReductionLoweringKind::Staged:
    return "hierarchical";
  }
  return "unknown";
}

void TreeReduceV2Pass::emitPrecisionCost(const ReductionRewritePlan &plan) {
  if (!emitPrecisionReport)
    return;
  const int64_t dimR = plan.dimR;
  const int64_t chainBound = dimR;
  const int64_t treeBound = treeDepthBound(dimR > 2 ? dimR : 2);
  llvm::errs() << "[precision-pricing] strategy="
               << stringifyReductionLoweringKind(plan.loweringKind)
               << " class=" << structureClass(plan.loweringKind)
               << " dimR=" << dimR << " dimA=" << plan.dimA
               << " vecLen=" << plan.vectorLength
               << " stages=" << plan.stages.size()
               << " | A_max(chain)<=" << chainBound
               << " A_max(tree)<=" << treeBound;
  if (treeBound > 0)
    llvm::errs() << " price(chain/tree)="
                 << (static_cast<double>(chainBound) / treeBound);
  llvm::errs() << "\\n";
}

LogicalResult
TreeReduceV2Pass::executeRewrite(IRRewriter &rewriter,
                                 const ReductionRewritePlan &plan) {
  Location loc = plan.rewriteLoop->getLoc();
  rewriter.setInsertionPoint(plan.rewriteLoop);
  emitPrecisionCost(plan);
'''


def patch(path, pairs):
    text = path.read_text()
    for anchor, add in pairs:
        if anchor not in text:
            sys.exit(f"ANCHOR NOT FOUND in {path}:\n{anchor[:200]}")
        if text.count(anchor) != 1:
            sys.exit(f"ANCHOR NOT UNIQUE ({text.count(anchor)}x) in {path}")
        text = text.replace(anchor, add, 1)
    path.write_text(text)
    print(f"patched {path}")


patch(TD, [(td_anchor, td_anchor + td_add)])
patch(CPP, [(inc_anchor, inc_add), (decl_anchor, decl_anchor + decl_add),
            (impl_anchor, impl_add)])
print("ok")
