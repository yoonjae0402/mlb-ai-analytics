"use client";
import { useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import { useWalkthrough } from "@/hooks/useWalkthrough";
import walkthroughSteps from "@/lib/walkthrough-steps";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

export default function GuidedWalkthrough() {
  const { isActive, currentStep, totalSteps, next, prev, skip } =
    useWalkthrough();
  const pathname = usePathname();
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);

  const step = walkthroughSteps[currentStep];

  const updateRect = useCallback(() => {
    if (!isActive || !step) return;
    if (pathname !== step.page) return;

    // Small delay to let page render
    const timer = setTimeout(() => {
      const el = document.querySelector(step.selector);
      if (el) {
        setTargetRect(el.getBoundingClientRect());
      } else {
        setTargetRect(null);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [isActive, step, pathname]);

  useEffect(() => {
    return updateRect();
  }, [updateRect]);

  // Update on scroll/resize
  useEffect(() => {
    if (!isActive) return;
    const handler = () => {
      if (!step || pathname !== step.page) return;
      const el = document.querySelector(step.selector);
      if (el) setTargetRect(el.getBoundingClientRect());
    };
    window.addEventListener("resize", handler);
    window.addEventListener("scroll", handler, true);
    return () => {
      window.removeEventListener("resize", handler);
      window.removeEventListener("scroll", handler, true);
    };
  }, [isActive, step, pathname]);

  if (!isActive || !step) return null;

  // While navigating to the target page
  if (pathname !== step.page) {
    return (
      <div className="fixed inset-0 z-[100] bg-black/60 flex items-center justify-center">
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-6 text-center">
          <p className="text-sm text-mlb-muted">Navigating...</p>
        </div>
      </div>
    );
  }

  const padding = 8;
  const tooltipWidth = 320;

  // Calculate tooltip position
  let tooltipStyle: React.CSSProperties = {
    position: "fixed",
    width: tooltipWidth,
    zIndex: 101,
  };

  if (targetRect) {
    const pos = step.position || "bottom";
    if (pos === "bottom") {
      tooltipStyle.top = targetRect.bottom + padding + 8;
      tooltipStyle.left = Math.max(
        16,
        Math.min(
          targetRect.left + targetRect.width / 2 - tooltipWidth / 2,
          window.innerWidth - tooltipWidth - 16
        )
      );
    } else if (pos === "top") {
      tooltipStyle.bottom = window.innerHeight - targetRect.top + padding + 8;
      tooltipStyle.left = Math.max(
        16,
        Math.min(
          targetRect.left + targetRect.width / 2 - tooltipWidth / 2,
          window.innerWidth - tooltipWidth - 16
        )
      );
    } else if (pos === "right") {
      tooltipStyle.top = targetRect.top + targetRect.height / 2 - 60;
      tooltipStyle.left = targetRect.right + padding + 8;
    } else {
      tooltipStyle.top = targetRect.top + targetRect.height / 2 - 60;
      tooltipStyle.right = window.innerWidth - targetRect.left + padding + 8;
    }
  } else {
    // Fallback: center
    tooltipStyle.top = "50%";
    tooltipStyle.left = "50%";
    tooltipStyle.transform = "translate(-50%, -50%)";
  }

  // SVG clip-path for the backdrop cutout
  const svgCutout = targetRect ? (
    <svg className="fixed inset-0 w-full h-full z-[100] pointer-events-none">
      <defs>
        <mask id="walkthrough-mask">
          <rect width="100%" height="100%" fill="white" />
          <rect
            x={targetRect.left - padding}
            y={targetRect.top - padding}
            width={targetRect.width + padding * 2}
            height={targetRect.height + padding * 2}
            rx={8}
            fill="black"
          />
        </mask>
      </defs>
      <rect
        width="100%"
        height="100%"
        fill="rgba(0,0,0,0.6)"
        mask="url(#walkthrough-mask)"
        className="pointer-events-auto"
        onClick={skip}
      />
    </svg>
  ) : (
    <div
      className="fixed inset-0 z-[100] bg-black/60"
      onClick={skip}
    />
  );

  return (
    <>
      {svgCutout}
      <div style={tooltipStyle} className="pointer-events-auto">
        <div className="bg-mlb-card border border-mlb-blue/40 rounded-xl p-4 shadow-2xl">
          <div className="flex items-start justify-between mb-2">
            <span className="text-[10px] text-mlb-blue font-semibold uppercase tracking-wider">
              Step {currentStep + 1} of {totalSteps}
            </span>
            <button
              onClick={skip}
              className="text-mlb-muted hover:text-mlb-text transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <h4 className="text-sm font-semibold text-mlb-text mb-1">
            {step.title}
          </h4>
          <p className="text-xs text-mlb-muted leading-relaxed mb-4">
            {step.description}
          </p>
          <div className="flex items-center justify-between">
            <button
              onClick={prev}
              disabled={currentStep === 0}
              className="flex items-center gap-1 text-xs text-mlb-muted hover:text-mlb-text disabled:opacity-30 transition-colors"
            >
              <ChevronLeft className="w-3 h-3" />
              Back
            </button>
            <button
              onClick={skip}
              className="text-xs text-mlb-muted hover:text-mlb-text transition-colors"
            >
              Skip Tour
            </button>
            <button
              onClick={next}
              className="flex items-center gap-1 text-xs bg-mlb-blue text-white px-3 py-1 rounded-lg hover:bg-mlb-blue/80 transition-colors"
            >
              {currentStep === totalSteps - 1 ? "Finish" : "Next"}
              {currentStep < totalSteps - 1 && (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
