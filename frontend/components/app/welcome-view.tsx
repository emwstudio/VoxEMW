import { PhoneIcon } from '@phosphor-icons/react';
import { LiangziAvatar } from '@/components/app/liangzi-avatar';

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

/**
 * 来电页：良子邀请你语音唠嗑，点绿色接听键进通话。
 */
export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="relative flex h-svh flex-col items-center justify-center bg-neutral-950 text-center">
        {/* 头像 + 呼吸光圈（来电感） */}
        <div className="relative grid place-items-center">
          <span className="absolute size-40 animate-ping rounded-full bg-amber-500/15 [animation-duration:2.4s]" />
          <span className="absolute size-48 rounded-full border border-amber-500/20" />
          <LiangziAvatar className="size-32 text-6xl" />
        </div>

        <p className="mt-12 text-3xl font-semibold text-white">良子</p>
        <p className="mt-3 text-sm text-neutral-400">邀请你语音唠嗑</p>

        {/* 接听键 */}
        <div className="mt-14 flex flex-col items-center gap-3">
          <button
            type="button"
            onClick={onStartCall}
            aria-label={startButtonText}
            className="grid size-20 place-items-center rounded-full bg-emerald-500 text-white shadow-[0_0_40px_-5px] shadow-emerald-500/50 transition-transform duration-200 hover:scale-105 hover:bg-emerald-400"
          >
            <PhoneIcon size={34} weight="fill" />
          </button>
          <span className="text-sm text-neutral-400">{startButtonText}</span>
        </div>
      </section>
    </div>
  );
};
