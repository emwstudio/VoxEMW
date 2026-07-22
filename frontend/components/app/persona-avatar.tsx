'use client';

import { useState } from 'react';
import { cn } from '@/lib/shadcn/utils';

interface PersonaAvatarProps {
  /** 缺省文字牌（如「良」「峰」），有照片时被 /{image} 覆盖 */
  char: string;
  /** public/ 下的照片路径，如 /liangzi.jpg；404 时回退文字牌 */
  image: string;
  /** 文字牌底色渐变 */
  gradient?: string;
  className?: string;
  style?: React.CSSProperties;
}

/** 角色头像：照片优先（放 frontend/public/ 即可），缺省退化为文字圆牌。 */
export function PersonaAvatar({ char, image, gradient, className, style }: PersonaAvatarProps) {
  const [photoOk, setPhotoOk] = useState(true);

  return (
    <div
      style={style}
      className={cn(
        'relative overflow-hidden rounded-full bg-gradient-to-br shadow-lg',
        gradient ?? 'from-amber-500 to-orange-700',
        className
      )}
    >
      <span className="absolute inset-0 grid place-items-center font-bold text-white select-none">
        {char}
      </span>
      {photoOk && image && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={image}
          alt={char}
          className="absolute inset-0 h-full w-full object-cover"
          onError={() => setPhotoOk(false)}
        />
      )}
    </div>
  );
}
