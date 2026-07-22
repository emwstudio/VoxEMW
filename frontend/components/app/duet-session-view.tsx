'use client';

import { Fragment, useEffect, useRef, useState } from 'react';
import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import { useLocalParticipant, useRoomContext, useTracks } from '@livekit/components-react';
import {
  ChatTextIcon,
  MicrophoneIcon,
  MicrophoneSlashIcon,
  PhoneDisconnectIcon,
} from '@phosphor-icons/react';
import { AgentAudioVisualizerAura } from '@/components/agents-ui/agent-audio-visualizer-aura';
import { PersonaAvatar } from '@/components/app/persona-avatar';
import { cn } from '@/lib/shadcn/utils';

type SpeakerKey = 'liangzi' | 'fengge' | 'user';

interface DuetLine {
  speaker: SpeakerKey;
  text: string;
}

const PERSONA: Record<
  SpeakerKey,
  {
    display: string;
    char: string;
    image: string;
    color: `#${string}`;
    gradient: string;
    side?: string;
  }
> = {
  liangzi: {
    display: '良子',
    char: '良',
    image: '/liangzi.jpg',
    color: '#1FD5F9',
    gradient: 'from-amber-500 to-orange-700',
    side: '捧场派',
  },
  fengge: {
    display: '峰哥',
    char: '峰',
    image: '/fengge.jpg',
    color: '#F5A623',
    gradient: 'from-sky-600 to-indigo-800',
    side: '锐评派',
  },
  user: {
    display: '你',
    char: '你',
    image: '',
    color: '#34D399',
    gradient: 'from-emerald-500 to-teal-700',
  },
};

/**
 * 峰哥×良子 双唠围观页：双头像 + 活跃发言者高亮 + 星云随发言人换色。
 * 后端经 data channel 推 {"speaker","text"} 驱动界面状态。
 */
export function DuetSessionView({
  interactive = false,
  topic,
  ref,
  className,
}: React.ComponentProps<'section'> & { interactive?: boolean; topic?: string }) {
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const [chatOpen, setChatOpen] = useState(false);
  const [active, setActive] = useState<SpeakerKey | null>(null);
  const [lines, setLines] = useState<DuetLine[]>([]);
  const [showTitle, setShowTitle] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 三方模式：自动开麦，用户随时插话
  useEffect(() => {
    if (interactive) {
      void localParticipant.setMicrophoneEnabled(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interactive]);

  // 音频轨：良子/峰哥吃 agent 远端轨（轮到他时才驱动对应星云）
  // 音频轨：良子/峰哥吃 agent 远端轨（轮到他时才驱动对应星云）
  const micTracks = useTracks([Track.Source.Microphone], { onlySubscribed: false });
  const agentTrack = micTracks.find((t) => !t.participant.isLocal);

  useEffect(() => {
    const handler = (payload: Uint8Array) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload)) as {
          speaker?: string;
          text?: string;
          phase?: string;
          topic?: string;
        };
        // 亮题仪式：全屏大标题 3s（无音效）
        if (msg.phase === 'topic_reveal' && msg.topic) {
          setShowTitle(msg.topic);
          window.setTimeout(() => setShowTitle(null), 3000);
          return;
        }
        if (msg.speaker === 'liangzi' || msg.speaker === 'fengge' || msg.speaker === 'user') {
          setActive(msg.speaker);
          if (msg.text)
            setLines((prev) => [...prev, { speaker: msg.speaker as SpeakerKey, text: msg.text! }]);
        } else {
          setActive(null);
        }
      } catch {
        // 非双唠消息，忽略
      }
    };
    room.on('dataReceived', handler);
    return () => {
      room.off('dataReceived', handler);
    };
  }, [room]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [lines]);

  const micEnabled = localParticipant.isMicrophoneEnabled;

  return (
    <section
      ref={ref}
      className={cn('relative h-full w-full overflow-hidden bg-neutral-950', className)}
    >
      {/* 顶部状态（发言状态栏已按需求移除） */}
      <div className="absolute inset-x-0 top-0 z-20 flex flex-col items-center pt-12">
        {topic && (
          <p className="mb-4 rounded-full border border-teal-400/40 bg-teal-400/10 px-7 py-2.5 text-lg">
            <span className="text-neutral-400">你的困惑</span>
            <span className="ml-2 font-semibold text-teal-300">「{topic}」</span>
          </p>
        )}
      </div>

      {/* 亮题仪式：全屏大标题 3s */}
      <AnimatePresence>
        {showTitle && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.35, ease: 'easeOut' }}
            className="absolute inset-0 z-40 grid place-items-center bg-neutral-950/85 backdrop-blur-sm"
          >
            <div className="flex -translate-y-16 flex-col items-center gap-5 px-8 text-center">
              <span className="text-2xl tracking-[0.5em] text-teal-400">今日解惑</span>
              <span className="text-5xl font-black text-white drop-shadow-[0_0_30px_rgba(45,212,191,0.5)] md:text-6xl">
                「{showTitle}」
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 中部：正反双方对擂，星云包头像，中间 PK */}
      <div className="absolute inset-0 z-10 flex items-center justify-center gap-24 pb-28 md:gap-36">
        {(['fengge', 'liangzi'] as const).map((key, idx) => {
          const p = PERSONA[key];
          const isActive = active === key;
          return (
            <Fragment key={key}>
              {idx === 1 && (
                <span className="text-6xl font-black tracking-widest text-rose-500 drop-shadow-[0_0_24px_rgba(244,63,94,0.6)] md:text-7xl">
                  PK
                </span>
              )}
              <div className="flex flex-col items-center">
                {/* 星云只围头像 */}
                <div className="relative grid place-items-center">
                  <AgentAudioVisualizerAura
                    size="xl"
                    state={isActive ? 'speaking' : 'listening'}
                    themeMode="dark"
                    color={p.color}
                    audioTrack={isActive ? agentTrack : undefined}
                    className={cn(
                      'transition-opacity duration-300',
                      isActive ? 'opacity-100' : 'opacity-40'
                    )}
                  />
                  <PersonaAvatar
                    char={p.char}
                    image={p.image}
                    gradient={p.gradient}
                    style={
                      isActive
                        ? { boxShadow: `0 0 60px 10px ${p.color}66, 0 0 0 5px ${p.color}` }
                        : undefined
                    }
                    className={cn(
                      'absolute size-28 text-4xl transition-all duration-300 md:size-36 md:text-5xl',
                      isActive ? 'scale-110' : 'opacity-70 ring-1 ring-white/10'
                    )}
                  />
                </div>
                {/* 辩论座席牌：名字+立场一体 */}
                <div
                  className={cn(
                    'mt-7 rounded-xl px-7 py-3 shadow-lg transition-all duration-300',
                    isActive ? 'scale-105' : 'opacity-60'
                  )}
                  style={{ backgroundColor: p.color }}
                >
                  <span className="text-2xl font-bold tracking-wider text-neutral-950">
                    {p.display} · {p.side}
                  </span>
                </div>
              </div>
            </Fragment>
          );
        })}
      </div>

      {/* 转写面板 */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 24 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="absolute inset-x-4 top-20 bottom-36 z-30 mx-auto max-w-xl overflow-hidden rounded-3xl border border-white/10 bg-black/70 backdrop-blur-md"
          >
            <div ref={scrollRef} className="flex h-full flex-col gap-3 overflow-y-auto p-4">
              {lines.length === 0 && (
                <p className="my-auto text-center text-sm text-neutral-500">
                  还没开聊，等良子起头…
                </p>
              )}
              {lines.map((line, i) => {
                const p = PERSONA[line.speaker];
                const isLiangzi = line.speaker === 'liangzi';
                const isUser = line.speaker === 'user';
                return (
                  <div
                    key={i}
                    className={cn(
                      'flex',
                      isUser ? 'justify-center' : isLiangzi ? 'justify-start' : 'justify-end'
                    )}
                  >
                    <div
                      className={cn(
                        'max-w-[80%] rounded-2xl px-4 py-2 text-sm leading-6',
                        isUser
                          ? 'bg-emerald-900/70 text-emerald-50'
                          : isLiangzi
                            ? 'rounded-tl-sm bg-cyan-950/80 text-cyan-50'
                            : 'rounded-tr-sm bg-amber-950/80 text-amber-50'
                      )}
                    >
                      <span className="mr-2 font-semibold" style={{ color: p.color }}>
                        {p.display}
                      </span>
                      {line.text}
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 底部控制钮 */}
      <div className="absolute inset-x-0 bottom-0 z-20 flex items-center justify-center gap-8 pb-12">
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => void localParticipant.setMicrophoneEnabled(!micEnabled)}
            aria-label={micEnabled ? '静音' : '取消静音'}
            className={cn(
              'grid size-16 place-items-center rounded-full transition-colors',
              micEnabled
                ? 'bg-white/10 text-white hover:bg-white/20'
                : 'bg-white text-neutral-900 hover:bg-neutral-200'
            )}
          >
            {micEnabled ? <MicrophoneIcon size={28} /> : <MicrophoneSlashIcon size={28} />}
          </button>
          <span className="text-sm text-neutral-500">{micEnabled ? '静音' : '已静音'}</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => room.disconnect()}
            aria-label="挂断"
            className="grid size-20 place-items-center rounded-full bg-red-500 text-white transition-colors hover:bg-red-400"
          >
            <PhoneDisconnectIcon size={34} />
          </button>
          <span className="text-sm text-neutral-500">挂断</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => setChatOpen((v) => !v)}
            aria-label="聊天记录"
            className={cn(
              'grid size-16 place-items-center rounded-full transition-colors',
              chatOpen
                ? 'bg-white text-neutral-900 hover:bg-neutral-200'
                : 'bg-white/10 text-white hover:bg-white/20'
            )}
          >
            <ChatTextIcon size={28} />
          </button>
          <span className="text-sm text-neutral-500">记录</span>
        </div>
      </div>
    </section>
  );
}
