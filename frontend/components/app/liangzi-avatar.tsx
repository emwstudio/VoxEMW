'use client';

import { useState } from 'react';
import { cn } from '@/lib/shadcn/utils';

interface LiangziAvatarProps {
  className?: string;
}

/**
 * 良子头像：优先显示 /liangzi.jpg（把照片放到 frontend/public/liangzi.jpg 即可），
 * 没有照片时退化为暖色「良」字圆牌。
 */
export function LiangziAvatar({ className }: LiangziAvatarProps) {
  const [photoOk, setPhotoOk] = useState(true);

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-full bg-gradient-to-br from-amber-500 to-orange-700 shadow-lg',
        className
      )}
    >
      <span className="absolute inset-0 grid place-items-center font-bold text-white select-none">
        良
      </span>
      {photoOk && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src="/liangzi.jpg"
          alt="良子"
          className="absolute inset-0 h-full w-full object-cover"
          onError={() => setPhotoOk(false)}
        />
      )}
    </div>
  );
}
