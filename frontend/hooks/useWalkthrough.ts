"use client";
import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import { useRouter, usePathname } from "next/navigation";
import walkthroughSteps from "@/lib/walkthrough-steps";
import React from "react";

interface WalkthroughContextType {
  isActive: boolean;
  currentStep: number;
  totalSteps: number;
  start: () => void;
  next: () => void;
  prev: () => void;
  skip: () => void;
  hasCompleted: boolean;
}

const WalkthroughContext = createContext<WalkthroughContextType>({
  isActive: false,
  currentStep: 0,
  totalSteps: walkthroughSteps.length,
  start: () => {},
  next: () => {},
  prev: () => {},
  skip: () => {},
  hasCompleted: false,
});

const STORAGE_KEY = "walkthrough_completed";

export function WalkthroughProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isActive, setIsActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [hasCompleted, setHasCompleted] = useState(false);

  useEffect(() => {
    setHasCompleted(localStorage.getItem(STORAGE_KEY) === "true");
  }, []);

  const finish = useCallback(() => {
    setIsActive(false);
    setCurrentStep(0);
    setHasCompleted(true);
    localStorage.setItem(STORAGE_KEY, "true");
  }, []);

  const start = useCallback(() => {
    setCurrentStep(0);
    setIsActive(true);
    const firstPage = walkthroughSteps[0].page;
    if (pathname !== firstPage) {
      router.push(firstPage);
    }
  }, [pathname, router]);

  const next = useCallback(() => {
    const nextIdx = currentStep + 1;
    if (nextIdx >= walkthroughSteps.length) {
      finish();
      return;
    }
    const nextPage = walkthroughSteps[nextIdx].page;
    if (pathname !== nextPage) {
      router.push(nextPage);
    }
    setCurrentStep(nextIdx);
  }, [currentStep, pathname, router, finish]);

  const prev = useCallback(() => {
    const prevIdx = Math.max(0, currentStep - 1);
    const prevPage = walkthroughSteps[prevIdx].page;
    if (pathname !== prevPage) {
      router.push(prevPage);
    }
    setCurrentStep(prevIdx);
  }, [currentStep, pathname, router]);

  const skip = useCallback(() => {
    finish();
  }, [finish]);

  const value = {
    isActive,
    currentStep,
    totalSteps: walkthroughSteps.length,
    start,
    next,
    prev,
    skip,
    hasCompleted,
  };

  return React.createElement(
    WalkthroughContext.Provider,
    { value },
    children
  );
}

export function useWalkthrough() {
  return useContext(WalkthroughContext);
}
