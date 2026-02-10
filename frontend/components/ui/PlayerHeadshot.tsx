"use client";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface PlayerHeadshotProps {
  mlbId: number;
  name: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeMap = {
  sm: "w-10 h-10 text-sm",
  md: "w-14 h-14 text-base",
  lg: "w-20 h-20 text-lg",
};

const imgSizeMap = {
  sm: 40,
  md: 56,
  lg: 80,
};

function getInitials(name: string) {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export default function PlayerHeadshot({
  mlbId,
  name,
  size = "sm",
  className,
}: PlayerHeadshotProps) {
  const [error, setError] = useState(false);

  const src = `https://img.mlb.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${mlbId}/headshot/67/current`;

  if (error || !mlbId) {
    return (
      <div
        className={cn(
          "rounded-full bg-mlb-surface flex items-center justify-center font-bold text-mlb-blue flex-shrink-0",
          sizeMap[size],
          className
        )}
      >
        {getInitials(name)}
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={name}
      width={imgSizeMap[size]}
      height={imgSizeMap[size]}
      onError={() => setError(true)}
      className={cn(
        "rounded-full object-cover flex-shrink-0 bg-mlb-surface",
        sizeMap[size],
        className
      )}
    />
  );
}
