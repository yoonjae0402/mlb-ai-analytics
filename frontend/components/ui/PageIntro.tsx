"use client";
import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface PageIntroProps {
  title: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
  pageKey: string;
}

export default function PageIntro({ title, children, icon, pageKey }: PageIntroProps) {
  const storageKey = `pageIntro_${pageKey}`;
  const [dismissed, setDismissed] = useState(true); // start hidden to avoid flash

  useEffect(() => {
    const stored = localStorage.getItem(storageKey);
    setDismissed(stored === "dismissed");
  }, [storageKey]);

  const handleDismiss = () => {
    setDismissed(true);
    localStorage.setItem(storageKey, "dismissed");
  };

  return (
    <AnimatePresence>
      {!dismissed && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.25 }}
          className="overflow-hidden"
        >
          <div className="bg-mlb-card border border-mlb-border border-l-4 border-l-mlb-blue rounded-xl p-5 mb-6 relative">
            <button
              onClick={handleDismiss}
              className="absolute top-3 right-3 text-mlb-muted hover:text-mlb-text transition-colors"
              aria-label="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-start gap-3 pr-6">
              {icon && <div className="text-mlb-blue mt-0.5 flex-shrink-0">{icon}</div>}
              <div>
                <h2 className="text-sm font-semibold text-mlb-text mb-1">{title}</h2>
                <div className="text-xs text-mlb-muted leading-relaxed space-y-1">
                  {children}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
