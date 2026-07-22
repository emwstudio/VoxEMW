'use client';

import { useState } from 'react';
import { BoxingGloveIcon } from '@phosphor-icons/react';
import { PersonaAvatar } from '@/components/app/persona-avatar';

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: (topic: string) => void;
}

/**
 * 来电页（录制版式）：良子 × 峰哥 两人辩论局。
 * 输入今日辩题，点「开始辩论」开杠。
 */
export const WelcomeView = ({
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  const [topic, setTopic] = useState('');

  return (
    <div ref={ref}>
      <section className="relative flex h-svh flex-col items-center justify-center bg-neutral-950 pt-12 text-center">
        {/* 双头像 + 呼吸光圈 */}
        <div className="relative grid place-items-center">
          <span className="absolute size-64 animate-ping rounded-full bg-teal-500/15 [animation-duration:2.4s]" />
          <span className="absolute size-72 rounded-full border border-teal-500/20" />
          <div className="flex items-center gap-10">
            <PersonaAvatar
              char="良"
              image="/liangzi.jpg"
              gradient="from-amber-500 to-orange-700"
              className="size-36 text-6xl shadow-[0_0_40px_-5px] shadow-cyan-400/40 md:size-40 md:text-7xl"
            />
            <PersonaAvatar
              char="峰"
              image="/fengge.jpg"
              gradient="from-sky-600 to-indigo-800"
              className="size-36 text-6xl shadow-[0_0_40px_-5px] shadow-amber-400/40 md:size-40 md:text-7xl"
            />
          </div>
        </div>

        <p className="mt-20 text-4xl font-semibold text-white md:text-5xl">良子 × 峰哥</p>
        <p className="mt-5 text-base text-neutral-400 md:text-lg">
          两人解惑局 · 良子捧场 × 峰哥锐评
        </p>

        {/* 辩题输入（加大） */}
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onStartCall(topic.trim())}
          placeholder="请输入你的困惑"
          maxLength={60}
          className="mt-16 w-[26rem] rounded-full border border-white/15 bg-white/5 px-7 py-4 text-center text-lg text-white placeholder-neutral-500 outline-none focus:border-teal-400/60 md:w-[30rem] md:py-5 md:text-xl"
        />

        <div className="mt-16 flex flex-col items-center gap-4">
          <button
            type="button"
            onClick={() => onStartCall(topic.trim())}
            aria-label="开始辩论"
            className="flex items-center gap-3 rounded-full bg-rose-600 px-10 py-5 text-xl font-semibold text-white shadow-[0_0_50px_-5px] shadow-rose-500/50 transition-transform duration-200 hover:scale-105 hover:bg-rose-500"
          >
            <BoxingGloveIcon size={30} weight="fill" />
            开始解惑
          </button>
        </div>
      </section>
    </div>
  );
};
