-- Initial schema for a NEW clothes table. Review/migrate existing tables separately.
create table public.clothes (
    id uuid primary key,
    user_id uuid not null references auth.users(id),
    name text not null,
    category text not null check (category in ('top', 'bottom', 'outer')),
    source_path text not null,
    image_bucket text not null,
    image_path text not null,
    image_sha256 text not null,
    image_mime_type text not null,
    image_width integer not null check (image_width > 0),
    image_height integer not null check (image_height > 0),
    color text,
    pattern text,
    is_available boolean not null default false,
    storage_location text,
    location_verified_at timestamptz,
    created_at timestamptz not null default now(),
    unique(user_id, source_path),
    unique(image_bucket, image_path)
);

alter table public.clothes enable row level security;
grant select on public.clothes to authenticated;
create policy "Users read own clothes" on public.clothes
    for select to authenticated using ((select auth.uid()) = user_id);

-- Private bucket. The importer uses a backend secret/service_role key.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('clothes-images', 'clothes-images', false, 6291456,
        array['image/jpeg', 'image/png', 'image/webp'])
on conflict (id) do nothing;

create policy "Users read own clothing images" on storage.objects
    for select to authenticated
    using (bucket_id = 'clothes-images'
           and (storage.foldername(name))[1] = (select auth.uid())::text);

alter table public.clothes
    add column if not exists description text,
    add column if not exists attributes jsonb,
    add column if not exists analysis_status text
        not null default 'pending',
    add column if not exists analyzed_image_sha256 text,
    add column if not exists analysis_model text,
    add column if not exists analyzed_at timestamptz,
    add column if not exists analysis_error text;